# Claude Code guidance

See [CONTRIBUTING.md](CONTRIBUTING.md) for all development practices — install, format, lint, test, and project structure. That file is the source of truth for both human contributors and AI assistants.

## Code Design

### Local helpers are private
When defining a helper function within a module, if that function will only be used within that same module or in tests,
use the `_` prefix in the function name.

### Outer functions precede inner functions
When defining composite functions within a module, the outermost functions should be on top, with the inner functions
(possibly private helpers) defined directly beneath of in the end of the file.

Example: If `f(x)` uses `g(x)` internally, define `f` first, than `g`.

## Git Workflow

### Do not commit or push without explicit instruction
Never commit or push changes on your own initiative after completing a task. Wait for the user to review the changes locally and explicitly ask you to commit (and separately to push). The exception is when the user's request already includes the instruction to commit and/or push (e.g. "implement X and push").

### Discuss non-obvious design decisions before implementing
When a request involves a trade-off where one approach is meaningfully more correct than another, surface the analysis and give a recommendation before carrying it out. One short paragraph is enough. Don't second-guess clear-cut requests, but don't implement ambiguous design choices blindly either.

This includes any change that alters the semantics or statistical properties of a metric — for example, switching from exact computation to a sampled approximation to solve a memory problem. Even if the approximation is statistically justifiable, the decision belongs to the user.
