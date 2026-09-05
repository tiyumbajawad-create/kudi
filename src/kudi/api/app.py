"""FastAPI serving layer. Real routers (ingest, transactions, anomalies,
recurring, insights) land in M5; this stub provides health/version so
`make serve` works from M0 onward."""

from importlib.metadata import version as pkg_version

from fastapi import FastAPI

app = FastAPI(title="Kudi API", version=pkg_version("kudi"))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def get_version() -> dict[str, str]:
    return {"version": pkg_version("kudi")}
