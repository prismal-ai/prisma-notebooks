"""Tiny helpers to author prisma-notebooks notebooks as Python data."""
from __future__ import annotations
import json, textwrap
from pathlib import Path

NEST = """# Enable nested event loop to use `await` directly in Jupyter.
import sys
if 'nest_asyncio' not in sys.modules:
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        %pip install -q nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()"""

def _src(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _src(text)}

def header(title, source, api, summary, phase_doc=None, extra_rows=()):
    src_link = (f"> Notebook generated from [`{source}`](https://github.com/prismal-ai/prismal/blob/main/{source})."
                if source.startswith("examples/") else
                f"> Notebook written against [`{source}`](https://github.com/prismal-ai/prismal/blob/main/{source}) (no standalone example in `examples/`).")
    rows = [("**API key**", api), ("**Source**", f"`{source}`")]
    if phase_doc: rows.append(("**Guide**", f"[`{phase_doc}`](https://github.com/prismal-ai/prismal/blob/main/{phase_doc})"))
    rows += list(extra_rows)
    table = "| Field | Value |\n|------|-------|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    return md(f"# {title}\n\n{src_link}\n\n{table}\n\n{textwrap.dedent(summary).strip()}")

def write(path, cells):
    nb = {"cells": [cells[0], code(NEST), *cells[1:]],
          "metadata": {"kernelspec": {"display_name": "Python 3 (prismal)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.13"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
