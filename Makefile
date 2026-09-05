.PHONY: install test lint typecheck data demo serve clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

data:
	python -m datagen.generate --seed 42 --out data/

demo: data
	python -m kudi.cli ingest data/generated/*.csv
	python -m kudi.cli report

serve:
	uvicorn kudi.api.app:app --reload --port 8000

clean:
	rm -rf data/generated .pytest_cache .mypy_cache .ruff_cache **/__pycache__
