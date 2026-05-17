
### Install
```bash
uv sync --frozen
```

### Format & Lint
```bash
uv run ruff format
uv run ruff check --fix
```

### Test
```bash
uv run pytest ./src/tests/
```

### Running Evaluations
TVD per iteration:
```bash
uv run python -m povs.eval.scripts.tvd_per_iter
```
