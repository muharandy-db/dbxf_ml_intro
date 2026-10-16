"""Build the complete notebooks in notebooks/ from the module markdown in modules/.

The markdown is the single source of truth: every ```python block becomes a code
cell, and the prose between code blocks becomes a %md cell. Other fenced blocks
(yaml, bash, text, mermaid ...) stay inside the markdown cells as-is.

Convention: ```python = a runnable notebook cell; ```py = a code *snippet* shown for
reading or partial editing (highlighted on GitHub, but never extracted as a cell).

Usage:
    python scripts/build_notebooks.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ROOT / "modules"
NOTEBOOKS = ROOT / "notebooks"

HEADER = "# Databricks notebook source"
SEP = "\n\n# COMMAND ----------\n\n"
FENCE = re.compile(r"^```(\w*)\s*$")


def split_cells(markdown: str):
    """Yield ("md" | "code", text) tuples in document order."""
    md, code, in_code, in_other = [], [], False, False
    for line in markdown.splitlines():
        m = FENCE.match(line)
        if in_code:
            if line.strip() == "```":
                yield "code", "\n".join(code)
                code, in_code = [], False
            else:
                code.append(line)
            continue
        if m and not in_other and m.group(1) == "python":
            if any(l.strip() for l in md):
                yield "md", "\n".join(md)
            md, in_code = [], True
            continue
        if in_other and line.strip() == "```":
            in_other = False
        elif m and not in_other:
            in_other = True
        md.append(line)
    if any(l.strip() for l in md):
        yield "md", "\n".join(md)


def clean_md(text: str) -> str:
    # <details>/<summary> does not render in notebooks: show solutions inline.
    text = re.sub(r"<summary>(.*?)</summary>", r"**\1**", text)
    text = re.sub(r"</?details>", "", text)
    # Module-to-module links point at .md files; keep the text, drop the link.
    text = re.sub(r"\[([^\]]+)\]\((?:\.\./|\./)?[^)]*\.md[^)]*\)", r"\1", text)
    return text.strip("\n")


def to_md_cell(text: str) -> str:
    lines = ["%md"] + clean_md(text).splitlines()
    return "\n".join(f"# MAGIC {l}".rstrip() for l in lines)


def to_code_cell(text: str) -> str:
    text = text.strip("\n")
    if text.lstrip().startswith("%"):
        # Magic cells (%pip, %sql ...) must be fully MAGIC-prefixed.
        return "\n".join(f"# MAGIC {l}".rstrip() for l in text.splitlines())
    return text


def build(md_path: Path) -> Path:
    cells = []
    for kind, text in split_cells(md_path.read_text()):
        if kind == "md":
            if clean_md(text).strip():
                cells.append(to_md_cell(text))
        else:
            cells.append(to_code_cell(text))
    out = NOTEBOOKS / (md_path.stem.replace("-", "_") + ".py")
    out.write_text(HEADER + "\n\n" + SEP.join(cells) + "\n")
    return out


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    for md_path in sorted(MODULES.glob("*.md")):
        out = build(md_path)
        n_code = sum(1 for k, _ in split_cells(md_path.read_text()) if k == "code")
        print(f"{md_path.name:45s} -> {out.relative_to(ROOT)}  ({n_code} code cells)")


if __name__ == "__main__":
    main()
