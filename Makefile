.PHONY: install install-dev test lint typecheck format qa \
	frontend-install frontend-dev frontend-build \
	serve dev dist clean

# Python
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
	uv run ruff format src/ tests/

qa: test lint typecheck format

# Frontend
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# Server
serve:
	uv run dsp-graph serve

dev:
	uv run dsp-graph serve --reload

# Distribution
dist: frontend-build
	uv build

clean:
	rm -rf dist/ build/ src/*.egg-info src/dsp_graph/__pycache__ \
		src/dsp_graph/static/ frontend/node_modules/ \
		.pytest_cache .mypy_cache
