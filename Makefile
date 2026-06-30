.PHONY: test precommit benchmark_core benchmark_aux

check_dirs := examples tests sppo

test:
	uv run pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	uv run pre-commit run --all-files

benchmark_core:
	bash ./benchmark/benchmark_core.sh

benchmark_aux:
	bash ./benchmark/benchmark_aux.sh

build:
	uv build

sync-torch260-vllm:
	uv sync --group torch260-vllm

sync-torch260-sglang:
	uv sync --group torch260-sglang
