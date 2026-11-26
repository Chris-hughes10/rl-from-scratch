# Converting a notebook into a Medium article

`tools/notebook_to_medium.py` turns a tutorial notebook into:

- an HTML page that can be pasted into Medium
- a numbered folder of images

It uses the notebook's **saved outputs**, so nothing is re-run. Run the notebook first if its outputs are stale.

## 1. Tag the code cells

Each code cell needs one tag in its metadata, which decides how it appears in the article:

| Tag | In the article | Use it for |
|---|---|---|
| `medium-code` | the source as a code block, followed by an image of its output (if any) | cells that **define a function or class**, because readers want to read and copy the code |
| `medium-screenshot` | one image of the source and output together | cells that **just run something**, such as calling a function, printing, plotting or building a table |
| `medium-skip` | left out | setup or housekeeping cells, such as freeing the GPU |

The script lists any untagged code cells and treats them as `medium-screenshot`.

**Exceptions to the rule:** cells that define a helper but exist to show an output should be tagged `medium-screenshot`. In `grpo_manual.ipynb`, this is the appendix's step 2 table, which defines `weights_as_sum` only to display the table.

**After editing the notebook,** tag any new code cells. You don't need to re-tag existing ones, because the tags live in each cell's metadata and are kept when the cell is edited. To tag every untagged code cell by the rule above, run this from the repo root:

```python
import ast, json
path = "tutorial/series/grpo_manual.ipynb"
nb = json.load(open(path))
for cell in nb["cells"]:
    tags = cell["metadata"].setdefault("tags", [])
    if cell["cell_type"] != "code" or any(tag.startswith("medium-") for tag in tags):
        continue
    defines = any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.parse("".join(cell["source"])).body)
    tags.append("medium-code" if defines else "medium-screenshot")
json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
open(path, "a").write("\n")
```

Then check the new tags against the exceptions above.

## 2. Run the script

```bash
uv run tools/notebook_to_medium.py tutorial/series/grpo_manual.ipynb \
    --site-url https://chris-hughes10.github.io/posts/grpo-from-scratch/ --code-url <REPO URL>
```

- **`--site-url`:** adds an italic line under the subtitle saying that the article was first published on the website, where the code can be copied and the tables and equations render properly, with a link to the notebook on GitHub (`--code-url`; a `REPO-TBC` placeholder until the repo is public).
- **On Medium:** set the story's canonical link to the same site URL.

- **Dependencies:** they're declared inside the script (PEP 723), so `uv` installs them in a separate environment and the project's `.venv` is not touched.
- **First time on a machine:** install Chromium for Playwright with `uv run --with playwright==1.49.1 playwright install chromium`.
- **Run time:** about 35 s for `grpo_manual.ipynb`.
- **MathJax:** equations are typeset by MathJax, which loads from a CDN, so the machine needs internet access. The script warns if MathJax doesn't load.

Output goes to `outputs/medium/<notebook name>/`, which is git-ignored:

| File | What it's for |
|---|---|
| `article.html` | The article to paste into Medium. Each image is a bold placeholder, e.g. **[IMAGE 14: 14_cell_c032.png — alt text: …]**. |
| `preview.html` | The same article with the images in place, for checking it. |
| `images/` | `01_….png`, `02_….png`, … in article order. The folder is rebuilt on every run, so the numbers stay correct after edits. |
| `_render.html` | The notebook's HTML that the screenshots were taken from. Useful for debugging. |

The script also lists relative links, such as `rl_systems.ipynb`, which won't work on Medium. Replace them with GitHub URLs after pasting.

## 3. Paste into Medium

The paste must come from the **rendered page in a browser**. Copying the HTML source pastes raw tags.

On the VM, serve the output folder and open it through VS Code's port forwarding:

```bash
cd outputs/medium/grpo_manual && python3 -m http.server 8765 --bind 127.0.0.1
```

1. Open `http://localhost:8765/article.html` in the local browser, select all, copy, and paste into a new Medium story.
2. Download `images/` (in VS Code's explorer, right-click the folder and choose **Download…**).
3. Drag each image onto its placeholder. Use `preview.html` to check what goes where.

## What the script handles, and why

Each of these came up while converting `grpo_manual.ipynb`, so keep them when changing the script:

- **Medium has no tables and no LaTeX.** Markdown tables and `$$…$$` equations become images. Inline math isn't handled, because the notebooks write symbols like ρ and π as Unicode.
- **Medium splits a pasted code block at its first empty line.** Every empty line inside a `<pre>` gets a single space. This was tested in Medium: `<pre><code>` and plain `<pre>` both work with spaces, but `<br>` line breaks don't. Don't use a non-breaking space, because it breaks Python for readers who copy the code.
- **Warnings and progress bars (stderr) are left out**, because they're noise in an article. Pass `--keep-stderr` to keep them.
- **Screenshots are made readable at Medium's width (about 700 px).**
  - The "In [n]:" prompts are hidden.
  - Long lines wrap instead of being cut off.
  - Images are 2× resolution.
  - Output-only images are cropped to the width of their content, so short outputs don't shrink.
- **Spacing:** extra margins around cells, tables and equations stop the 12 px padding from catching neighbouring content.
- **Medium has two heading sizes**, so `###` and `####` both become the smaller one.

## If something breaks

- **`'MathBlockParser' object has no attribute 'parse_axt_heading'`:** nbconvert 7.16.4 doesn't work with mistune ≥ 3.1. Keep `mistune==3.0.2` pinned in the script header.
- **Timeout waiting for `.MJXc-display > .MathJax_CHTML`:** nbconvert's MathJax output has changed. Open `_render.html` and look at how a display equation is rendered (MathJax 2 CommonHTML uses `span.MJXc-display > span.MathJax_CHTML`), then update the `equation` selector in `take_screenshots`.
- **A screenshot shows the wrong cell:** the script finds cells by their `id="cell-id=…"` anchor in nbconvert's HTML. Check that every cell in the notebook has an `id`; nbformat ≥ 4.5 adds one.
