# chess-v2

### Quickstart
```bash
# Create and sync the virtual environment from pyproject
uv sync

# Activate the virtual environment (ensures `which python` points to the venv)
source .venv/bin/activate

# Install runtime deps
uv pip install megatron-core[mlm]
uv pip install --no-build-isolation transformer-engine[pytorch]

# Clone Megatron-LM and install from source with the active interpreter
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.13.1
uv pip install .
uv pip install psutil
uv pip install 'numpy<2.0.0'

# Build the C++ dataset helper against the same interpreter (fixes python3 vs python mismatches)
cd megatron/core/datasets
rm -f helpers_cpp*.so
make PYTHON="$(which python)"
cd ../../../..

bash ../scripts/apex.sh
```

### Common uv commands
- Install a runtime dependency: `uv add requests`
- Install a dev dependency: `uv add -d ruff`
- Run a module: `uv run python -m your_module`
- Update dependencies: `uv lock --upgrade && uv sync`

### License
MIT
