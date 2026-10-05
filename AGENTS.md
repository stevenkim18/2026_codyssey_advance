# Development Environment

- Use the single uv-managed environment in the project root.
- Target Python 3.13, as declared in `pyproject.toml`.
- Run Python commands with `uv run`, for example `uv run python script.py`.
- Add or remove dependencies with `uv add` and `uv remove`; do not use `pip` directly.
- Keep `uv.lock` synchronized and commit it together with `pyproject.toml`.
