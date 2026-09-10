"""In-process background job manager for video analysis.

Video processing takes far longer than one HTTP request, so uploads run in a
background thread and the browser polls for progress. The original code kept a
bare module-level dict that grew forever and had no concurrency cap, no
cancellation, and no cleanup. This adds a small, thread-safe manager with:

* a typed job lifecycle (processing / done / error / cancelled),
* a maximum number of concurrent jobs (further submissions are rejected so the
  box can't be swamped),
* cooperative cancellation via an event the worker polls,
* progress reporting, and
* lazy expiration of finished jobs after a max age.

Deliberately dependency-free — no Celery/Redis. Suitable for a single-process
deployment; not a distributed queue.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from modules.logging_setup import get_logger

log = get_logger("jobs")


class JobStatus:
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


_ACTIVE = {JobStatus.PROCESSING}
_FINISHED = {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}


@dataclass
class Job:
    id: str
    type: str
    status: str = JobStatus.PROCESSING
    done: int = 0
    total: int = 0
    result: object = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, max_concurrent: int = 2, max_age_s: float = 3600.0):
        self.max_concurrent = max_concurrent
        self.max_age_s = max_age_s
        self._jobs: dict[str, Job] = {}
        self._active = 0
        self._lock = threading.Lock()

    def submit(self, job_type: str, target) -> str | None:
        """Start ``target(progress_cb, cancel_check) -> result`` in a background
        thread. Returns the job id, or None if the concurrency cap is reached."""
        self.cleanup()
        with self._lock:
            if self._active >= self.max_concurrent:
                return None
            job = Job(id=uuid.uuid4().hex, type=job_type)
            self._jobs[job.id] = job
            self._active += 1
        threading.Thread(
            target=self._run, args=(job, target), daemon=True
        ).start()
        return job.id

    def _run(self, job: Job, target) -> None:
        def progress(done, total):
            with self._lock:
                job.done, job.total = done, total
                job.updated_at = time.time()

        def cancel_check():
            return job.cancel_event.is_set()

        log.info("job %s (%s) started", job.id, job.type)
        try:
            result = target(progress, cancel_check)
            with self._lock:
                if job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    log.info("job %s cancelled", job.id)
                else:
                    job.status = JobStatus.DONE
                    job.result = result
                    log.info("job %s done (%s frames)", job.id, job.done)
                job.updated_at = time.time()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the poller
            with self._lock:
                job.status = JobStatus.ERROR
                job.error = str(exc)
                job.updated_at = time.time()
            log.exception("job %s failed: %s", job.id, exc)
        finally:
            with self._lock:
                self._active -= 1

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status not in _ACTIVE:
                return False
            job.cancel_event.set()
            return True

    def active_count(self) -> int:
        with self._lock:
            return self._active

    def cleanup(self) -> None:
        """Drop finished jobs older than ``max_age_s`` (lazy, called on submit)."""
        now = time.time()
        with self._lock:
            for jid in list(self._jobs):
                job = self._jobs[jid]
                if job.status in _FINISHED and now - job.updated_at > self.max_age_s:
                    del self._jobs[jid]
