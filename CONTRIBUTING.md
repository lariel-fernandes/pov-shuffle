
### Install
```bash
uv sync --frozen
```
- Always required before testing/evaluating if there were changes to C++/CUDA sources.

### Format & Lint
```bash
uv run --no-sync ruff format
uv run --no-sync ruff check --fix
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
