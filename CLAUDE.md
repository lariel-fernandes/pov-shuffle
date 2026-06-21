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
