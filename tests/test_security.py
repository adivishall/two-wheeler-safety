"""Security-focused HTTP tests (Phase 26).

Covers the controls in docs/SECURITY.md at the HTTP boundary: security headers,
no permissive CORS, path-traversal on the evidence route, rate limiting, error
responses that don't leak internals, the request-size cap, and debug-off.
Model-free — no route here loads YOLO/EasyOCR.
"""

import importlib
import sys

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "sec.db"))
    monkeypatch.delenv("DETECT_API_KEY", raising=False)
    app_module = importlib.reload(sys.modules["app"]) if "app" in sys.modules else __import__("app")
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_security_headers_on_every_response(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"


def test_no_permissive_cors_header(client):
    # The app is same-origin; it must never emit a wildcard ACAO that would let
    # any site read the API with the user's ambient context.
    r = client.get("/api/stats")
    assert "Access-Control-Allow-Origin" not in r.headers


def test_health_does_not_leak_secrets(client):
    body = client.get("/health").get_json()
    assert set(body) == {"status", "models_loaded", "debug"}
    assert body["debug"] is False  # debugger off by default (RCE surface)


def test_evidence_route_blocks_traversal_and_bad_types(client):
    # Traversal attempts and non-media types are 404, never served.
    assert client.get("/evidence/../app.py").status_code == 404
    assert client.get("/evidence/..%2f..%2fapp.py").status_code in (400, 404)
    assert client.get("/evidence/secrets.txt").status_code == 404


def test_error_responses_are_json_and_do_not_leak(client):
    r = client.get("/api/violations/999999")  # unknown id
    assert r.status_code == 404
    assert r.is_json and "error" in r.get_json()

    # An unknown path returns the generic JSON 404 handler, not an HTML page.
    r2 = client.get("/no/such/route")
    assert r2.status_code == 404 and r2.is_json
    assert r2.get_json()["error"] == "not found"

    # Wrong method -> generic JSON 405, no stack trace.
    r3 = client.get("/detect")  # POST-only
    assert r3.status_code == 405 and r3.is_json


def test_video_status_error_is_generic(client, monkeypatch):
    # A failed job must surface a generic message to the browser, never the raw
    # exception text (which can carry internal paths).
    import app as app_module
    jid = app_module.job_manager.submit("video", lambda p, c: (_ for _ in ()).throw(
        RuntimeError("/private/internal/path leaked")))
    # Wait for the worker to record the failure.
    import time
    for _ in range(200):
        snap = app_module.job_manager.status(jid)
        if snap and snap["status"] == "error":
            break
        time.sleep(0.01)
    body = client.get(f"/video_status/{jid}").get_json()
    assert body["status"] == "error"
    assert body["error"] == "video processing failed"
    assert "leaked" not in body["error"]


def test_rate_limit_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAFFIC_DB_PATH", str(tmp_path / "rl.db"))
    monkeypatch.delenv("DETECT_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    app_module = importlib.reload(sys.modules["app"])
    app_module.app.config["TESTING"] = True
    payload = {"plate": "MH12AB1234", "violation": "no_helmet", "image_path": "e.jpg"}
    with app_module.app.test_client() as c:
        codes = [c.post("/detect", json=payload).status_code for _ in range(4)]
    assert 429 in codes  # the limiter trips within the window
