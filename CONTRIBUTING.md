
## Tasks
optimization:
- [ ] take care of the leftover TODOs in optim and cuda code (e.g. filling tables of optimized device parameters, limits, etc.)

testing:
- [ ] review unit tests with inferred gpu block size
- [ ] extend unit tests to inferred algorithm options
- [ ] consider using xorshift32 in numpy workers to have reproducibility of the CUDA implementation (might also need to switch the host-side rng engine)

evaluation:
- [ ] review new evaluation scripts
- [ ] adjust time per deck size evaluation to use inferred algorithm options when not specified, while documenting the inferred in the report
- [ ] do some plot that compares the resulting TVD and other biases against a standard local block shuffle!
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
│   │   ├── utils.py         # Generic stateless utilities, math, etc
│   │   ├── common.py        # Shared use-case-aware logic and utilities
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
uv run --no-sync bash -c "ruff format && ruff check --fix"
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
SRC_DIR=lib PROGRAM=povs make
```
- By default, this compiles and runs the `main` function of [`../__cuda/lib/povs.cu`](./src/povs/__cuda/lib/povs.cu)
- For running specific test files, use `SRC_DIR=test` and `PROGRAM=file_name`
- See `make help` for other options.
