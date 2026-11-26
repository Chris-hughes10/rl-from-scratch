"""Copy a tutorial notebook into the website repo as a Quarto post, ready to render from its saved outputs.

The post lives in <site>/myblog/posts/<slug>/:
  index.ipynb     this notebook, minus its title and subtitle lines (Quarto shows them from the front matter), minus
                  warnings and other stderr output, and with a "run it yourself" link to the code added at the top
  images/         the images the notebook's markdown uses
  _metadata.yml   the front matter (title, subtitle, date, draft, ...). Written only if it doesn't exist yet, so edits
                  made in the website repo survive re-syncing. Quarto applies a directory's _metadata.yml to the
                  documents in it.

Nothing is executed: the site's CI only has Quarto, and .ipynb files render from their saved outputs. Re-run the
notebook first if its outputs are stale.

Usage: python tools/notebook_to_quarto.py tutorial/series/grpo_manual.ipynb --slug grpo-from-scratch \
           [--site ../Chris-hughes10.github.io] [--code-url URL]
"""

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path

DEFAULT_SITE = Path(__file__).resolve().parents[2] / "Chris-hughes10.github.io"
CODE_URL_PLACEHOLDER = "https://github.com/Chris-hughes10/REPO-TBC"  # TODO: the public repo isn't live yet

FRONT_MATTER = """\
title: "{title}"
subtitle: "{subtitle}"
date: "{date}"
categories: ["Reinforcement Learning", LLMs, PyTorch]
description: "TODO: one or two sentences for previews and search; Quarto shows it under the subtitle, so don't repeat it"
reading-time: {reading_time}
draft: true   # rendered, but not listed or linked anywhere (the site uses draft-mode: unlinked); remove to publish
"""


def split_title(first_cell_source: str) -> tuple[str, str, str]:
    """'# Title\\n\\n*Subtitle*\\n...' -> (title, subtitle, the rest of the cell)."""
    match = re.match(r"#\s+(?P<title>.+)\n+\*(?P<subtitle>[^*\n]+)\*\n+(?P<rest>.*)", first_cell_source, flags=re.S)
    if not match:
        raise ValueError("expected the first cell to start with '# Title', a blank line, then '*Subtitle*'")
    return match["title"].strip(), match["subtitle"].strip(), match["rest"]


LOG_LINE = re.compile(r"^(DEBUG|INFO|WARNING|ERROR) \d\d-\d\d \d\d:\d\d:\d\d \[")  # vLLM's logger, e.g. "WARNING 11-16 14:14:33 [config.py:70] ..."


def drop_log_lines(output: dict) -> dict | None:
    """None for stderr output; stdout without vLLM's log lines (None if nothing is left); anything else unchanged."""
    if output["output_type"] != "stream":
        return output
    if output.get("name") == "stderr":
        return None
    lines = [line for line in "".join(output["text"]).splitlines(keepends=True) if not LOG_LINE.match(line)]
    return {**output, "text": lines} if "".join(lines).strip() else None


LIST_ITEM = re.compile(r"^([-*+]|\d+\.)\s")


def add_blank_lines_before_blocks(source: str) -> tuple[str, int]:
    """Pandoc (unlike Jupyter) only starts a list or a table after a blank line; without one, the items run into the
    paragraph above. Insert the missing blank lines, outside code blocks. Returns the new source and how many were added."""
    lines, out, in_code, added = source.split("\n"), [], False, 0
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        previous = out[-1] if out else ""
        starts_block = LIST_ITEM.match(line) or line.startswith("|")
        continues_block = LIST_ITEM.match(previous) or previous.startswith(("|", " ", "\t"))
        if not in_code and starts_block and previous.strip() and not continues_block:
            out.append("")
            added += 1
        out.append(line)
    return "\n".join(out), added


def reading_time_minutes(notebook: dict) -> int:
    words = sum(len(re.findall(r"\w+", "".join(cell["source"]))) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    code_lines = sum("".join(cell["source"]).count("\n") + 1 for cell in notebook["cells"] if cell["cell_type"] == "code")
    return round(words / 230 + code_lines / 60, -1) or 5  # ~230 words a minute for prose, skimming ~60 lines of code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--slug", required=True, help="the post's folder name, e.g. grpo-from-scratch")
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE, help="the website repo")
    parser.add_argument("--code-url", default=CODE_URL_PLACEHOLDER, help="where readers can get the notebook")
    parser.add_argument("--date", default=None, help="post date for a new _metadata.yml (default: today)")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text())
    post_dir = args.site.resolve() / "myblog" / "posts" / args.slug
    post_dir.mkdir(parents=True, exist_ok=True)

    # Title and subtitle move to the front matter
    first = notebook["cells"][0]
    title, subtitle, rest = split_title("".join(first["source"]))
    first["source"] = rest.splitlines(keepends=True)

    # A link to the code, as the first thing readers see
    notebook["cells"].insert(0, {
        "cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
        "source": [f"> **Run it yourself.** This article is a Jupyter notebook: the full code, with every output shown here, "
                   f"is [on GitHub]({args.code_url}).\n"],
    })

    # Warnings and progress bars are noise in an article: drop stderr, and vLLM's log lines, which it prints to stdout
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = [output for output in map(drop_log_lines, cell["outputs"]) if output is not None]

    # Lists and tables need a blank line before them in pandoc
    for cell_number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            fixed, added = add_blank_lines_before_blocks("".join(cell["source"]))
            if added:
                cell["source"] = fixed.splitlines(keepends=True)
                print(f"added {added} blank line(s) before a list or table in markdown cell {cell_number}; worth fixing in the notebook too")

    (post_dir / "index.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")

    # The images the markdown uses, at the same relative paths
    image_paths = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            image_paths.update(path for path in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", "".join(cell["source"])) if not re.match(r"https?://", path))
    for relative_path in sorted(image_paths):
        target = post_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(notebook_path.parent / relative_path, target)

    metadata_path = post_dir / "_metadata.yml"
    if metadata_path.exists():
        print(f"kept the existing {metadata_path} (title and subtitle in the notebook: {title!r} / {subtitle!r})")
    else:
        from datetime import date
        metadata_path.write_text(FRONT_MATTER.format(title=title.replace('"', '\\"'), subtitle=subtitle.replace('"', '\\"'),
                                                     date=args.date or date.today().isoformat(),
                                                     reading_time=int(reading_time_minutes(notebook))))
        print(f"wrote {metadata_path}")
    print(f"wrote {post_dir / 'index.ipynb'} and {len(image_paths)} images")
    if args.code_url == CODE_URL_PLACEHOLDER:
        print(f"note: the code link is a placeholder ({CODE_URL_PLACEHOLDER}); pass --code-url once the repo is public")


if __name__ == "__main__":
    main()
