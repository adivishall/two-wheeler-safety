"""Role-based auth (Phase 16) + the new payment/session/audit/trace endpoints.

Auth is disabled unless role keys are configured, so the default client keeps
the app fully open (backwards compatible); the authed client sets role keys and
checks that mutations are gated."""

import importlib
import sys

import pytest


def _make_client(monkeypatch, tmp_path, env=None):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.delenv("DETECT_API_KEY", raising=False)
    for k in ("VIEWER_API_KEYS", "REVIEWER_API_KEYS", "ADMIN_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if "app" in sys.modules:
        app_module = importlib.reload(sys.modules["app"])
    else:
        import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def client(monkeypatch, tmp_path):
    return _make_client(monkeypatch, tmp_path)


@pytest.fixture
def authed(monkeypatch, tmp_path):
    """A client with role keys configured (auth ON)."""
    return _make_client(monkeypatch, tmp_path, {
        "REVIEWER_API_KEYS": "rev-key",
        "ADMIN_API_KEYS": "adm-key",
    })


def _seed_violation(client):
    client.post("/detect", json={"plate": "MH12AB1234", "violation": "no_helmet",
                                 "image_path": "evidence/x.jpg"})
    return client.get("/api/violations").get_json()["items"][0]["id"]


# -- auth disabled by default -----------------------------------------------

def test_review_open_when_no_keys_configured(client):
    vid = _seed_violation(client)
    r = client.post(f"/api/violations/{vid}/review", json={"review_status": "confirmed"})
    assert r.status_code == 200


# -- auth enabled ------------------------------------------------------------

def test_review_requires_reviewer_role_when_enabled(authed):
    vid = _seed_violation(authed)
    # no key -> 403
    assert authed.post(f"/api/violations/{vid}/review",
                       json={"review_status": "confirmed"}).status_code == 403
    # reviewer key -> 200
    ok = authed.post(f"/api/violations/{vid}/review",
                     json={"review_status": "confirmed"},
                     headers={"X-API-Key": "rev-key"})
    assert ok.status_code == 200


def test_admin_key_satisfies_reviewer_requirement(authed):
    vid = _seed_violation(authed)
    ok = authed.post(f"/api/violations/{vid}/review",
                     json={"review_status": "dismissed"},
                     headers={"X-API-Key": "adm-key"})
    assert ok.status_code == 200


def test_audit_endpoint_requires_admin(authed):
    _seed_violation(authed)
    assert authed.get("/api/audit").status_code == 403
    assert authed.get("/api/audit",
                      headers={"X-API-Key": "rev-key"}).status_code == 403
    ok = authed.get("/api/audit", headers={"X-API-Key": "adm-key"})
    assert ok.status_code == 200
    assert any(e["event"] == "violation_created" for e in ok.get_json()["events"])


# -- payment lifecycle endpoint ---------------------------------------------

def test_payment_endpoint_transitions(client):
    vid = _seed_violation(client)
    ok = client.post(f"/api/violations/{vid}/payment", json={"status": "paid"})
    assert ok.status_code == 200
    detail = client.get(f"/api/violations/{vid}").get_json()
    assert detail["status"] == "paid"
    # paid is terminal -> illegal transition is a 400
    bad = client.post(f"/api/violations/{vid}/payment", json={"status": "unpaid"})
    assert bad.status_code == 400


def test_payment_endpoint_rejects_unknown_status(client):
    vid = _seed_violation(client)
    r = client.post(f"/api/violations/{vid}/payment", json={"status": "refunded"})
    assert r.status_code == 400


# -- audit + trace read endpoints (auth off) --------------------------------

def test_audit_open_when_no_keys(client):
    vid = _seed_violation(client)
    events = client.get("/api/audit").get_json()["events"]
    assert any(e["record_id"] == str(vid) for e in events)


def test_trace_endpoint_returns_empty_for_photo_fine(client):
    vid = _seed_violation(client)
    # photo/detect fines carry no detection trace -> empty list, 200
    r = client.get(f"/api/violations/{vid}/trace")
    assert r.status_code == 200
    assert r.get_json()["trace"] == []


def test_sessions_endpoint_lists(client):
    # no sessions yet -> empty list, 200
    assert client.get("/api/sessions").get_json()["sessions"] == []
    assert client.get("/api/sessions/nope").status_code == 404
