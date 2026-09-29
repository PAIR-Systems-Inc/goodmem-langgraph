.PHONY: all test test_live lint format type build check

UV_RUN = uv run --locked --all-groups --all-extras

all: check

check: lint type test build

# Offline graph and framework tests; no server or model credentials required.
test:
	$(UV_RUN) pytest --disable-socket --allow-unix-socket tests/unit_tests

# Creates and deletes temporary spaces on the explicitly configured server.
test_live:
	$(UV_RUN) pytest tests/integration_tests

lint:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff check --select I --fix .
	$(UV_RUN) ruff format .

type:
	$(UV_RUN) mypy goodmem_langgraph examples

build:
	uv build
	$(UV_RUN) pytest --disable-socket --allow-unix-socket tests/packaging_tests
