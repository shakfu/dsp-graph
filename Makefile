.PHONY: install install-dev test lint typecheck format qa clean \
		dist sdist check publish-test publish

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

clean:
	rm -rf dist/ build/ src/*.egg-info src/dsp_graph/__pycache__ .pytest_cache .mypy_cache

dist: clean
	uv build
	uv run twine check dist/*

sdist: clean
	uv build --sdist
	uv run twine check dist/*

check:
	uv run twine check dist/*

publish-test: dist
	uv run twine upload --repository testpypi dist/*

publish: dist
	uv run twine upload dist/*
