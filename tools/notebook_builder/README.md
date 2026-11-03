# notebook_builder

Source of the Level 14 (wave 2026-10) notebooks. Each `cells/<name>.py` defines `CELLS` with three helpers from
`nbkit.py`:

- `header(...)` — the cell 0 summary table;
- `md(...)` — a markdown cell;
- `code(...)` — a code cell.

The `nest_asyncio` cell is inserted automatically as cell 1.

```bash
uv run python tools/notebook_builder/build.py                  # all notebooks, no outputs
uv run python tools/notebook_builder/build.py --execute        # all notebooks, executed with outputs
uv run python tools/notebook_builder/build.py --execute typed_decisions
```

Edit the `cells/*.py` file, not the `.ipynb`, so a rebuild does not overwrite your change.
