"""Build (and optionally execute) the cell-defined notebooks under ``cells/``.

Each ``cells/<name>.py`` exposes ``CELLS`` (a list made with :mod:`nbkit`) and is
written to ``notebooks/<name>.ipynb``. With ``--execute`` every notebook is run
top to bottom with ``nbclient`` and saved with its outputs; any failing cell
fails the build.

Usage::

    uv run python tools/notebook_builder/build.py [--execute] [name ...]
"""

from __future__ import annotations

import argparse
import re
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path[:0] = [str(HERE), str(HERE / "cells")]

from nbkit import write  # noqa: E402


def _execute(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=180, resources={"metadata": {"path": str(path.parent)}}).execute()
    _clean(nb)
    nbformat.write(nb, path)


_NOISE = ("otel.disabled", "langfuse.disabled")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", re.MULTILINE)


def _clean(nb: object) -> None:
    """Keep outputs reproducible: drop stderr, startup log noise and run metadata."""
    for cell in nb.cells:  # type: ignore[attr-defined]
        cell.metadata.pop("execution", None)
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.outputs:
            if out.get("output_type") == "stream":
                if out.get("name") == "stderr":
                    continue
                text = _STAMP.sub("", _ANSI.sub("", out["text"]))
                out["text"] = "".join(
                    line for line in text.splitlines(keepends=True)
                    if not any(n in line for n in _NOISE)
                )
                if not out["text"]:
                    continue
            kept.append(out)
        cell.outputs = kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="notebook names (default: all in cells/)")
    parser.add_argument("--execute", action="store_true", help="run and save outputs")
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in (HERE / "cells").glob("*.py"))
    failed = []
    for name in names:
        target = NOTEBOOKS / f"{name}.ipynb"
        write(target, importlib.import_module(name).CELLS)
        if args.execute:
            try:
                _execute(target)
            except Exception as exc:  # noqa: BLE001 — report and keep going
                failed.append(name)
                print(f"FAIL {name}: {str(exc).strip().splitlines()[-1]}")
                continue
        print(f"ok   {target.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
