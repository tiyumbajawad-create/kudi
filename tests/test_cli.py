"""CLI smoke tests (§9: CLI drives the same service layer as the API)."""

from pathlib import Path

from typer.testing import CliRunner

from kudi.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_ingest_command(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = runner.invoke(app, ["ingest", str(FIXTURES / "chase_sample.csv"), "--db", str(db)])
    assert result.exit_code == 0
    assert "parsed=2" in result.stdout
    assert "Enriched 2 transactions" in result.stdout


def test_ingest_missing_file_errors_cleanly(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = runner.invoke(app, ["ingest", "/no/such/file.csv", "--db", str(db)])
    assert result.exit_code == 1


def test_report_command(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    runner.invoke(app, ["ingest", str(FIXTURES / "chase_sample.csv"), "--db", str(db)])
    result = runner.invoke(app, ["report", "--account-id", "chase_sample", "--db", str(db)])
    assert result.exit_code == 0
    assert "Income>Salary" in result.stdout


def test_report_unknown_account(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = runner.invoke(app, ["report", "--account-id", "nope", "--db", str(db)])
    assert result.exit_code == 0
    assert "No transactions found" in result.stdout


def test_anomalies_command(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    runner.invoke(app, ["ingest", str(FIXTURES / "chase_sample.csv"), "--db", str(db)])
    result = runner.invoke(
        app, ["anomalies", "--account-id", "chase_sample", "--min-score", "0.0", "--db", str(db)]
    )
    assert result.exit_code == 0


def test_subscriptions_command(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    runner.invoke(app, ["ingest", str(FIXTURES / "chase_sample.csv"), "--db", str(db)])
    result = runner.invoke(app, ["subscriptions", "--account-id", "chase_sample", "--db", str(db)])
    assert result.exit_code == 0
