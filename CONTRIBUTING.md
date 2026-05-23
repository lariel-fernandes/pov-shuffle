
### Install
```bash
uv sync --frozen
```
- Always required before testing/evaluating if there were changes to C++/CUDA sources.

### Format & Lint
Python:
```bash
uv run --no-sync ruff format
uv run --no-sync ruff check --fix
```

C++/CUDA:
```bash
find src -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
```

### Test
```bash
.venv/bin/python -m pytest ./src/tests/
```

### Running Evaluations
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
