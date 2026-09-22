.PHONY: install test lint typecheck data demo serve clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src datagen scripts

data:
	python -m datagen.generate --seed 42 --out data/

demo: data
	python -m kudi.cli ingest data/generated/chase_csv/*.csv --db data/kudi.db
	python -m kudi.cli report --account-id chase-checking --db data/kudi.db
	python -m kudi.cli anomalies --account-id chase-checking --min-score 0.9 --db data/kudi.db
	python -m kudi.cli subscriptions --account-id chase-checking --db data/kudi.db

serve:
	uvicorn kudi.api.app:app --reload --port 8000

clean:
	rm -rf data/generated data/kudi.db .pytest_cache .mypy_cache .ruff_cache **/__pycache__
