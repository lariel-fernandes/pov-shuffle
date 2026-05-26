
## Tasks
device code:
- [ ] Investigate sequence of zeroes after the first 16 elements
- [ ] Adjust variable names, comments and documentation to facilitate the mental model
cuda optimizations:
- [ ] check whether flat_comp_tiler is partitioning the computation evenly across threads
- [ ] Use vector copy for large instances
- [ ] Consider using a flat vector copy layout when instance size is 1
interface:
- [ ] implement heuristic to infer recommended algorithm options based on problem size and device properties
evaluation:
- [ ] do some plot that compares against a standard local block shuffle!
- [ ] add the exercise of breaking point by dataset size

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
│   │   ├── types.py         # Public API types and aliases
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

## Debug

### Running standalone CUDA programs
```bash
make
```
- See `make help` for more details.
