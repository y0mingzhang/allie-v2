# chess-v2

A minimal Python project scaffold configured for `uv`.

### Requirements
- uv

### Quickstart
```bash
# Create and sync the virtual environment from pyproject
uv sync

# Activate the virtual environment (POSIX)
source .venv/bin/activate

# Verify Python
uv run python -V

# Run tests
uv run pytest

# Lint & format with ruff
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy .
```

### Common uv commands
- Install a runtime dependency: `uv add requests`
- Install a dev dependency: `uv add -d ruff`
- Run a module: `uv run python -m your_module`
- Update dependencies: `uv lock --upgrade && uv sync`

### License
MIT
