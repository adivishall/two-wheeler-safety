import sys
import importlib

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
