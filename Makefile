.PHONY: install install-dev test lint typecheck format qa \
	frontend-install frontend-dev frontend-build \
	frontend-lint frontend-typecheck frontend-test frontend-qa \
	serve dev dist clean check publish publish-test

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

frontend-lint:
	cd frontend && npm run lint

frontend-typecheck:
	cd frontend && npx tsc --noEmit

frontend-test:
	cd frontend && npm test

# Frontend gate (lint + typecheck + tests); kept separate from `qa` so the
# Python-only flow does not require a Node toolchain.
frontend-qa: frontend-lint frontend-typecheck frontend-test

# Server
serve:
	uv run dsp-graph serve

dev:
	uv run dsp-graph serve --reload

# Distribution
dist: frontend-build
	uv build

check:
	uv run twine check dist/*

publish: dist check
	uv run twine upload dist/*

publish-test: dist check
	uv run twine upload --repository testpypi dist/*

clean:
	rm -rf dist/ build/ src/*.egg-info src/dsp_graph/__pycache__ \
		src/dsp_graph/static/ frontend/node_modules/ \
		.pytest_cache .mypy_cache
