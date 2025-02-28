mypy:
	uv pip install mypy
	uv run mypy src/deepweeds

.PHONY: mypy
