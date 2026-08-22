.PHONY: help install dev test test-unit test-integration lint format type-check check init-db seed ingest clean

PYTHON := python
PIP := pip
UVICORN := uvicorn
PYTEST := pytest
RUFF := ruff
MYPY := mypy
POWERSHELL := powershell -NoProfile -ExecutionPolicy Bypass -Command

help: ## Show this help message
	@echo "NL-to-SQL RAG Pipeline - Developer Commands"
	@echo "============================================"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies (including dev extras)
	$(PIP) install -e ".[dev]"

dev: ## Start the FastAPI dev server with hot-reload
	$(UVICORN) nl_to_sql.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload --log-level info

test: ## Run the full test suite
	$(PYTEST) tests/ -v

test-unit: ## Run only unit tests
	$(PYTEST) tests/unit/ -v

test-integration: ## Run only integration tests
	$(PYTEST) tests/integration/ -v

lint: ## Run ruff linter
	$(RUFF) check src/ tests/ scripts/

format: ## Auto-format code with ruff
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

type-check: ## Run mypy static type checker
	$(MYPY) src/nl_to_sql/

check: lint type-check ## Run all code quality checks

init-db: ## Initialize the PostgreSQL history database
	$(PYTHON) scripts/init_db.py

seed: ## Seed the sample e-commerce SQLite database
	$(PYTHON) scripts/seed_postgresql.py

ingest: ## Ingest DB schema into the vector store
	$(PYTHON) scripts/ingest_schema.py

clean: ## Remove generated caches, coverage output, and frontend build artifacts
	$(POWERSHELL) " \
	$$targets = @( \
		'.mypy_cache', '.pytest_cache', '.ruff_cache', '.venv', '__pycache__', 'htmlcov', \
		'../frontend/.vite', '../frontend/dist', '../frontend/node_modules' \
	); \
	Get-ChildItem -Path . -Recurse -Directory -Force | \
		Where-Object { $$_.Name -eq '__pycache__' } | \
		ForEach-Object { Remove-Item -LiteralPath $$_.FullName -Recurse -Force -ErrorAction SilentlyContinue }; \
	foreach ($$target in $$targets) { \
		if (Test-Path -LiteralPath $$target) { \
			Remove-Item -LiteralPath $$target -Recurse -Force -ErrorAction SilentlyContinue; \
		} \
	}; \
	Get-ChildItem -Path . -Recurse -File -Include '*.pyc','*.pyo','.coverage','coverage.xml' -Force -ErrorAction SilentlyContinue | \
		ForEach-Object { Remove-Item -LiteralPath $$_.FullName -Force -ErrorAction SilentlyContinue }; \
	Write-Host 'Cleaned generated artifacts.' \
	"
