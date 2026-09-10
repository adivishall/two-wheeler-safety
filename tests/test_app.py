import importlib
import sys

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Fresh Flask test client per test, pointed at a temp SQLite DB so
    tests never touch the real traffic.db. app.py reads TRAFFIC_DB_PATH
    and runs init_db() at import time, so the env var must be set before
    (re)importing it."""
    db_path = str(tmp_path / "test_traffic.db")
    monkeypatch.setenv("TRAFFIC_DB_PATH", db_path)
    monkeypatch.delenv("DETECT_API_KEY", raising=False)

    if "app" in sys.modules:
        app_module = importlib.reload(sys.modules["app"])
    else:
        import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_detect_requires_all_fields(client):
    response = client.post("/detect", json={"plate": "MH12AB1234"})
    assert response.status_code == 400


def test_detect_records_a_fine(client):
    response = client.post("/detect", json={
        "plate": "MH12AB1234",
        "violation": "no_helmet",
        "image_path": "evidence/x.jpg"
    })
    assert response.status_code == 200
    assert response.get_json()["message"] == "Violation Recorded"


@pytest.mark.parametrize("violation,expected_amount", [
    ("no_helmet", 500),
    ("triple_riding", 1000),
    ("overspeed", 700),
    ("some_unknown_violation", 300),
])
def test_fine_amounts_by_violation_type(client, violation, expected_amount):
    client.post("/detect", json={
        "plate": "MH12AB1234",
        "violation": violation,
        "image_path": "evidence/x.jpg"
    })
    data = client.get("/get_fines/MH12AB1234").get_json()
    assert data["fines"][0]["amount"] == expected_amount


def test_analyze_without_an_image_is_rejected(client):
    # The empty-upload guard runs before any model is loaded, so this
    # exercises the /analyze route without needing the YOLO weights.
    response = client.post("/analyze", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_video_without_a_file_is_rejected(client):
    # Same idea for the video route: the empty-upload guard runs before any
    # model loads or background thread starts.
    response = client.post("/analyze_video", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_video_status_unknown_job_is_404(client):
    response = client.get("/video_status/does-not-exist")
    assert response.status_code == 404


def test_plate_lookup_is_case_and_whitespace_insensitive(client):
    client.post("/detect", json={
        "plate": "  mh12ab1234  ",
        "violation": "no_helmet",
        "image_path": "evidence/x.jpg"
    })

    data = client.get("/get_fines/mh12ab1234").get_json()
    assert len(data["fines"]) == 1

    data = client.get("/get_fines/MH12AB1234").get_json()
    assert len(data["fines"]) == 1


def test_get_fines_for_unknown_plate_is_empty(client):
    data = client.get("/get_fines/NOSUCHPLATE").get_json()
    assert data["fines"] == []
    assert data["total"] == 0


def test_total_only_counts_unpaid_fines(client):
    client.post("/detect", json={
        "plate": "MH12AB1234", "violation": "no_helmet", "image_path": "e.jpg"
    })
    client.post("/detect", json={
        "plate": "MH12AB1234", "violation": "triple_riding", "image_path": "e.jpg"
    })

    data = client.get("/get_fines/MH12AB1234").get_json()
    assert data["total"] == 500 + 1000


def test_fines_endpoint_returns_labeled_objects(client):
    client.post("/detect", json={
        "plate": "MH12AB1234", "violation": "no_helmet", "image_path": "e.jpg"
    })

    data = client.get("/fines").get_json()
    assert len(data) == 1
    assert set(data[0].keys()) == {
        "id", "plate", "violation", "amount", "image_path", "timestamp", "status"
    }


def test_detect_rejects_wrong_or_missing_api_key_when_configured(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_traffic.db")
    monkeypatch.setenv("TRAFFIC_DB_PATH", db_path)
    monkeypatch.setenv("DETECT_API_KEY", "secret123")

    app_module = importlib.reload(sys.modules["app"]) if "app" in sys.modules else __import__("app")
    app_module.app.config["TESTING"] = True

    with app_module.app.test_client() as client:
        payload = {"plate": "MH12AB1234", "violation": "no_helmet", "image_path": "e.jpg"}

        response = client.post("/detect", json=payload)
        assert response.status_code == 401

        response = client.post("/detect", json=payload, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

        response = client.post("/detect", json=payload, headers={"X-API-Key": "secret123"})
        assert response.status_code == 200


# --- Phase 9: input validation & security ----------------------------------

def test_health_endpoint(client):
    data = client.get("/health").get_json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is False  # lazy; nothing uploaded yet


def test_detect_rejects_invalid_plate(client):
    r = client.post("/detect", json={
        "plate": "!!", "violation": "no_helmet", "image_path": "e.jpg"})
    assert r.status_code == 400


def test_detect_rejects_traversal_image_path(client):
    r = client.post("/detect", json={
        "plate": "MH12AB1234", "violation": "no_helmet",
        "image_path": "../../etc/passwd"})
    assert r.status_code == 400


def test_evidence_rejects_non_media_and_subpaths(client):
    assert client.get("/evidence/notes.txt").status_code == 404
    assert client.get("/evidence/sub/dir.jpg").status_code == 404


def test_analyze_rejects_unsupported_extension(client):
    from io import BytesIO
    data = {"image": (BytesIO(b"whatever"), "photo.txt")}
    r = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


def test_analyze_rejects_non_image_content(client):
    # A .jpg name but the bytes aren't an image -> magic-byte sniff rejects it
    # before the model is ever loaded.
    from io import BytesIO
    data = {"image": (BytesIO(b"this is not a jpeg"), "photo.jpg")}
    r = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "not a valid image" in r.get_json()["error"]


def test_analyze_video_rejects_unsupported_extension(client):
    from io import BytesIO
    data = {"video": (BytesIO(b"whatever"), "clip.txt")}
    r = client.post("/analyze_video", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


# --- dashboard analytics + filtering + review API --------------------------

def _record(client, plate="MH12AB1234", violation="no_helmet"):
    return client.post("/detect", json={
        "plate": plate, "violation": violation, "image_path": "evidence/x.jpg"})


def test_api_stats_reflects_real_data(client):
    _record(client, violation="no_helmet")
    _record(client, plate="KA05CD9", violation="triple_riding")
    data = client.get("/api/stats").get_json()
    assert data["total_violations"] == 2
    assert data["total_fines"] == 500 + 1000
    assert data["vehicles"] == 2
    assert data["by_review_status"] == {"pending": 2}
    assert len(data["recent"]) == 2


def test_api_violations_filter_by_type(client):
    _record(client, violation="no_helmet")
    _record(client, violation="overspeed")
    data = client.get("/api/violations?type=overspeed").get_json()
    assert data["total"] == 1
    assert data["items"][0]["type"] == "overspeed"


def test_api_violations_pagination_shape(client):
    for _ in range(5):
        _record(client)
    data = client.get("/api/violations?limit=2&offset=0").get_json()
    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["limit"] == 2 and data["offset"] == 0


def test_api_violation_detail_and_404(client):
    _record(client)
    vid = client.get("/api/violations").get_json()["items"][0]["id"]
    detail = client.get(f"/api/violations/{vid}").get_json()
    assert detail["id"] == vid
    assert "evidence" in detail
    assert client.get("/api/violations/999999").status_code == 404


def test_api_review_flow(client):
    _record(client)
    vid = client.get("/api/violations").get_json()["items"][0]["id"]

    # missing review_status -> 400
    assert client.post(f"/api/violations/{vid}/review", json={}).status_code == 400
    # unknown state -> 400
    bad = client.post(f"/api/violations/{vid}/review", json={"review_status": "nope"})
    assert bad.status_code == 400
    # valid -> 200 and reflected in the listing
    ok = client.post(f"/api/violations/{vid}/review",
                     json={"review_status": "confirmed", "notes": "clear plate"})
    assert ok.status_code == 200
    listed = client.get("/api/violations?review_status=confirmed").get_json()
    assert listed["total"] == 1
    # unknown id -> 404
    assert client.post("/api/violations/999999/review",
                       json={"review_status": "confirmed"}).status_code == 404


def test_video_cancel_unknown_job_is_404(client):
    assert client.post("/video_cancel/does-not-exist").status_code == 404


def test_rate_limit_returns_429_when_exceeded(monkeypatch, tmp_path):
    # A tiny per-minute cap makes the limiter trip deterministically. app.py
    # builds its limiter from config at import, so set the env then reload.
    db_path = str(tmp_path / "rl.db")
    monkeypatch.setenv("TRAFFIC_DB_PATH", db_path)
    monkeypatch.delenv("DETECT_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")

    app_module = importlib.reload(sys.modules["app"]) if "app" in sys.modules else __import__("app")
    app_module.app.config["TESTING"] = True

    with app_module.app.test_client() as c:
        payload = {"plate": "MH12AB1234", "violation": "no_helmet", "image_path": "e.jpg"}
        assert c.post("/detect", json=payload).status_code == 200
        assert c.post("/detect", json=payload).status_code == 200
        assert c.post("/detect", json=payload).status_code == 429  # 3rd exceeds cap
