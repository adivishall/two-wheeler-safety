"""JobManager tests (thread-based but synchronized with events, not sleeps)."""

import threading
import time

from modules.jobs import JobManager, JobStatus


def _wait_for(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_job_completes_and_reports_result():
    jm = JobManager(max_concurrent=2)

    def target(progress, cancel_check):
        progress(5, 10)
        return {"ok": True}

    jid = jm.submit("video", target)
    assert _wait_for(lambda: jm.status(jid)["status"] == JobStatus.DONE)
    assert jm.status(jid)["result"] == {"ok": True}


def test_job_error_is_captured():
    jm = JobManager()

    def target(progress, cancel_check):
        raise RuntimeError("boom")

    jid = jm.submit("video", target)
    assert _wait_for(lambda: jm.status(jid)["status"] == JobStatus.ERROR)
    assert "boom" in jm.status(jid)["error"]


def test_progress_is_reported():
    jm = JobManager()
    release = threading.Event()

    def target(progress, cancel_check):
        progress(3, 12)
        release.wait(2.0)
        return "done"

    jid = jm.submit("video", target)
    assert _wait_for(lambda: jm.status(jid)["done"] == 3)
    assert jm.status(jid)["total"] == 12
    release.set()


def test_max_concurrent_rejects_further_submissions():
    jm = JobManager(max_concurrent=1)
    gate = threading.Event()

    def blocking(progress, cancel_check):
        gate.wait(2.0)
        return "ok"

    first = jm.submit("video", blocking)
    assert first is not None
    assert _wait_for(lambda: jm.active_count() == 1)
    second = jm.submit("video", blocking)  # at capacity
    assert second is None
    gate.set()


def test_cancellation_is_cooperative():
    jm = JobManager()
    started = threading.Event()

    def cancellable(progress, cancel_check):
        started.set()
        for _ in range(1000):
            if cancel_check():
                return "partial"
            time.sleep(0.005)
        return "full"

    jid = jm.submit("video", cancellable)
    assert started.wait(2.0)
    assert jm.cancel(jid) is True
    assert _wait_for(lambda: jm.status(jid)["status"] == JobStatus.CANCELLED)


def test_unknown_job_status_is_none():
    jm = JobManager()
    assert jm.status("nope") is None
    assert jm.cancel("nope") is False


def test_cleanup_expires_finished_jobs():
    jm = JobManager(max_age_s=0.0)  # everything finished is immediately expiry-eligible

    def target(progress, cancel_check):
        return "done"

    jid = jm.submit("video", target)
    assert _wait_for(lambda: jm.status(jid) is not None
                     and jm.status(jid)["status"] == JobStatus.DONE)
    time.sleep(0.01)
    jm.cleanup()
    assert jm.status(jid) is None  # expired and dropped
