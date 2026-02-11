.PHONY: install install-dev test lint typecheck format qa clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy --strict src/

format:
	uv run ruff format src/ tests/ examples/

qa: test lint typecheck format
