.PHONY: install install-dev test lint typecheck format qa clean \
		dist sdist check publish-test publish \
		examples examples-gen examples-build \
		examples-gen-dsp build-examples-gen-dsp validate-examples-gen-dsp

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

EXAMPLES := $(wildcard examples/*.py)
CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable

examples-gen:
	@mkdir -p build
	@for f in $(EXAMPLES); do \
		echo "  GEN  $$f"; \
		uv run python "$$f" > /dev/null; \
	done

examples-build: examples-gen
	@for f in build/*.cpp; do \
		echo "  CXX  $$f"; \
		$(CXX) $(CXXFLAGS) -c "$$f" -o "$${f%.cpp}.o"; \
	done

examples: examples-build
	@echo "All examples generated and compiled."

examples-gen-dsp:
	@uv run python examples/gen_dsp_targets.py

build-examples-gen-dsp:
	@uv run python examples/gen_dsp_targets.py --build

validate-examples-gen-dsp:
	@uv run python examples/gen_dsp_targets.py --validate

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
