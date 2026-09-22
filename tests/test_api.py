"""API tests via httpx against the app in-process (design doc §11)."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("KUDI_DB_PATH", str(db_path))
    import importlib

    import kudi.api.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def _upload_chase_fixture(client: TestClient) -> None:
    content = (FIXTURES / "chase_sample.csv").read_bytes()
    files = [("files", ("chase-checking.csv", io.BytesIO(content), "text/csv"))]
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_version(client: TestClient) -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_ingest_infers_account_from_filename(client: TestClient) -> None:
    _upload_chase_fixture(client)
    resp = client.get("/transactions", params={"account_id": "chase-checking"})
    assert resp.status_code == 200
    txns = resp.json()
    assert len(txns) == 2
    assert all(t["account_id"] == "chase-checking" for t in txns)


def test_ingest_categorizes_transactions(client: TestClient) -> None:
    _upload_chase_fixture(client)
    resp = client.get("/transactions", params={"account_id": "chase-checking"})
    txns = resp.json()
    payroll = next(t for t in txns if "PAYROLL" in t["raw_description"])
    assert payroll["category"] == "Income>Salary"
    assert payroll["category_source"] == "rule"


def test_correction_persists_and_wins_over_rule(client: TestClient) -> None:
    _upload_chase_fixture(client)
    txns = client.get("/transactions", params={"account_id": "chase-checking"}).json()
    txn_id = txns[0]["txn_id"]

    resp = client.post(f"/transactions/{txn_id}/category", json={"category": "Other>Uncategorized"})
    assert resp.status_code == 200
    assert resp.json()["category"] == "Other>Uncategorized"
    assert resp.json()["category_source"] == "user"


def test_correction_on_unknown_txn_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/transactions/does-not-exist/category", json={"category": "Other>Uncategorized"}
    )
    assert resp.status_code == 404


def test_anomalies_endpoint_returns_list(client: TestClient) -> None:
    _upload_chase_fixture(client)
    resp = client.get("/anomalies", params={"account_id": "chase-checking", "min_score": 0.0})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_ack_unknown_txn_returns_404(client: TestClient) -> None:
    resp = client.post("/anomalies/does-not-exist/ack")
    assert resp.status_code == 404


def test_recurring_endpoint_shape(client: TestClient) -> None:
    _upload_chase_fixture(client)
    resp = client.get("/recurring", params={"account_id": "chase-checking"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"audit", "price_hikes", "missed_charges", "duplicate_billing"}


def test_insights_summary_shape(client: TestClient) -> None:
    _upload_chase_fixture(client)
    resp = client.get("/insights/summary", params={"account_id": "chase-checking"})
    assert resp.status_code == 200
    body = resp.json()
    assert "2026-01" in body
