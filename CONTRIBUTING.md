# Contributing

1. Fork + branch (`feat/...` or `fix/...`).
2. Install dev deps: `pip install -r requirements.txt pytest ruff`.
3. Run checks before opening a PR:
   - `python -m pytest tests/ -q` (all green)
   - `python -m compileall -q . -x 'node_modules|\.git'`
   - `ruff check .` (lint)
4. Open a PR with a clear description. Tests are mandatory for new modules.
