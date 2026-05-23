
## Tasks
python:
- [ ] Generalize code for algorithm pre-flight and seed selection
docs:
- [ ] Document how to run standalone CUDA program
build:
- [ ] Parameterize the instance size at build time with default of 1
device code:
- [ ] Define CUTE tensors in kernel, using lazy transform with the offset and wrap around logic
- [ ] Do vectorized copy and thread0 shuffling
- [ ] Implement Fisher-Yates shuffle of a CUTE tensor for the device

## Project Structure

```
.
├── CMakeLists.txt           # Not for builds, just for IDE integration 
├── MANIFEST.in              # Include CUDA extension sources in source distributions
├── pyproject.toml           # Project and tools configuration (UV, linters, etc)
├── setup.py                 # Build script for the CUDA extension
├── src
│   ├── povs
│   │   ├── __init__.py      # Public re-exports and user-facing API
│   │   ├── numpy.py         # NumPy interface
│   │   ├── torch.py         # PyTorch interface
│   │   ├── utils.py         # Stateless utilities
│   │   ├── options.py       # API options schema
│   │   ├── eval/...         # Benchmark and evaluation resources
│   │   │
│   │   ├── __cuda           # CUDA extension sources
│   │   │   ├── module.cpp   # Module definition
│   │   │   ├── binds/...    # Library bindings / interfacing
│   │   │   └── lib/...      # Standalone library / core implementation
│   │   │
│   │   ├── _cuda.cpython-*  # CUDA extension build artifact
│   │   └── _cuda.pyi        # CUDA extension stubs (importable from `povs._cuda`)
│   │
│   └── tests/...            # Python unit tests
│
└── uv.lock                  # Dependency lockfile for development and tests
```

## Install
```bash
uv sync --frozen
```
- Always required before testing/evaluating if there were changes to C++/CUDA sources.

## Format & Lint
Python:
```bash
uv run --no-sync ruff format
uv run --no-sync ruff check --fix
```

C++/CUDA:
```bash
find src -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
```

## Test
```bash
.venv/bin/python -m pytest ./src/tests/
```

## Running Evaluations
TVD per iteration:
```bash
.venv/bin/python -m povs.eval.scripts.tvd_per_iter
```

## Generate stubs
```bash
.venv/bin/python -c '
import torch
import pybind11_stubgen;

pybind11_stubgen.main(["povs._cuda", "-o", "src"]);
'
```
