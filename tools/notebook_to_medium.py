# /// script
# requires-python = ">=3.10"
# dependencies = ["nbconvert==7.16.4", "mistune==3.0.2", "playwright==1.49.1"]
# ///
"""Turn a tagged notebook into a Medium-ready article: prose, code blocks and numbered images.

Each code cell is tagged in its metadata:
  medium-code        the source becomes a code block in the article; its output (if any) becomes an image
  medium-screenshot  the source and output together become one image
  medium-skip        left out
Untagged code cells are treated as medium-screenshot (and listed, so you can tag them).

Medium has no tables and no LaTeX, so markdown tables and $$...$$ equations also become images, and images the notebook
already shows (![...](images/x.png)) are copied into the same numbered sequence. Everything uses the notebook's saved
outputs; nothing is re-run. Warnings (stderr output, and vLLM's log lines on stdout) are left out unless you pass --keep-stderr.

Output, in outputs/medium/<notebook name>/ (git-ignored):
  article.html   open in a browser, select all, copy, paste into a new Medium story; every image is a numbered placeholder
  preview.html   the same article with the images in place, for checking
  images/        01_....png, 02_....png, ... in article order: drag each onto its placeholder

Full workflow (tagging, pasting into Medium, troubleshooting): tools/notebook_to_medium.md

Usage:   uv run tools/notebook_to_medium.py tutorial/series/grpo_manual.ipynb [--out DIR]
First time only:   uv run --with playwright==1.49.1 playwright install chromium
"""

import argparse
import html
import re
import shutil
from pathlib import Path

import mistune
import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
TAGS = {"medium-code", "medium-screenshot", "medium-skip"}
GITHUB_REPO = "https://github.com/Chris-hughes10/rl-from-scratch"  # this repo, on GitHub (as in notebook_to_quarto.py)

# Applied to the notebook's HTML before screenshotting: no "In [3]:" prompts, and long lines wrap instead of being cut off
SCREENSHOT_CSS = """
.jp-InputPrompt, .jp-OutputPrompt, .jp-Collapser { display: none !important; }
.jp-Cell-inputWrapper pre, .jp-OutputArea-output pre { white-space: pre-wrap !important; word-break: break-word; }
.jp-OutputArea-output { overflow: visible !important; }
body, .jp-Notebook { background: white !important; }
.jp-Notebook { padding: 8px !important; }
/* More space around everything we screenshot than the padding we add, so no screenshot catches its neighbours */
.jp-Cell { margin: 40px 0 !important; }
.jp-Cell-outputWrapper { margin-top: 14px !important; }
.jp-RenderedMarkdown table, .MJXc-display { margin: 30px auto !important; }
"""
# Width from an element's left edge to the right edge of its content (text, images, tables)
CONTENT_WIDTH_JS = """element => {
    let right = 0;
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const range = document.createRange();
        range.selectNodeContents(node);
        for (const rect of range.getClientRects()) if (rect.width > 0) right = Math.max(right, rect.right);
    }
    for (const media of element.querySelectorAll("img, svg, table, canvas"))
        right = Math.max(right, media.getBoundingClientRect().right);
    const box = element.getBoundingClientRect();
    return right > box.left ? right - box.left : box.width;
}"""
VIEWPORT_WIDTH = 1000  # px; Medium shows images about 700 px wide, so wider screenshots get small text
PIXEL_RATIO = 2        # retina-quality images
PADDING = 12           # px of white space around each screenshot


def markdown_to_html(source: str) -> str:
    """Markdown → HTML, with each $$...$$ equation swapped for a marker that becomes an image placeholder."""
    source = re.sub(r"\$\$(.+?)\$\$", lambda match: "\n\n@@EQUATION@@\n\n", source, flags=re.S)
    return mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])(source)


class Article:
    """Collects the article's HTML and the list of images to make, in order."""

    def __init__(self, notebook_dir: Path, image_dir: Path):
        self.notebook_dir, self.image_dir = notebook_dir, image_dir
        self.parts: list[str] = []      # HTML, with {IMAGE n} where image n goes
        self.screenshots: list[dict] = []  # what to capture in the browser, and the file name
        self.images: list[str] = []     # file names, in article order (index = number - 1)
        self.alt_texts: dict[int, str] = {}

    def add_image(self, name: str, alt: str = "") -> str:
        number = len(self.images) + 1
        file_name = f"{number:02d}_{name}.png"
        self.images.append(file_name)
        if alt:
            self.alt_texts[number] = alt
        return f"{{IMAGE {number}}}"

    def add_screenshot(self, name: str, cell_id: str, kind: str, index: int = 0) -> str:
        placeholder = self.add_image(name)
        self.screenshots.append({"file": self.images[-1], "cell_id": cell_id, "kind": kind, "index": index})
        return placeholder

    def add_markdown(self, cell, cell_number: int):
        body = markdown_to_html(cell.source)
        counters = {"equation": 0, "table": 0}

        def replace(match):
            if match.group("equation"):
                counters["equation"] += 1
                return self.add_screenshot(f"equation_c{cell_number:03d}", cell.id, "equation", counters["equation"] - 1)
            if match.group("table"):
                counters["table"] += 1
                return self.add_screenshot(f"table_c{cell_number:03d}", cell.id, "table", counters["table"] - 1)
            src, alt = html.unescape(match.group("src")), html.unescape(match.group("alt") or "")
            if re.match(r"https?://", src):
                return match.group(0)  # a hosted image: Medium can fetch it itself
            placeholder = self.add_image(Path(src).stem, alt)
            shutil.copy(self.notebook_dir / src, self.image_dir / self.images[-1])
            return placeholder

        pattern = re.compile(r'(?P<equation><p>@@EQUATION@@</p>)|(?P<table><table>.*?</table>)'
                             r'|<img src="(?P<src>[^"]+)"(?: alt="(?P<alt>[^"]*)")?\s*/?>', re.S)
        self.parts.append(pattern.sub(replace, body))

    def add_code(self, cell, cell_number: int, tag: str):
        if tag == "medium-skip":
            return
        if tag == "medium-code":
            self.parts.append(f"<pre><code>{html.escape(cell.source)}</code></pre>")
            if cell.outputs:
                self.parts.append(f"<p>{self.add_screenshot(f'output_c{cell_number:03d}', cell.id, 'output')}</p>")
        else:
            self.parts.append(f"<p>{self.add_screenshot(f'cell_c{cell_number:03d}', cell.id, 'cell')}</p>")

    def render(self, with_images: bool) -> str:
        def image(match):
            number = int(match.group(1))
            file_name = self.images[number - 1]
            if with_images:
                return f'<img src="images/{file_name}" alt="{html.escape(self.alt_texts.get(number, ""))}">'
            alt = f" — alt text: {html.escape(self.alt_texts[number])}" if number in self.alt_texts else ""
            return f'<strong>[IMAGE {number:02d}: {file_name}{alt}]</strong>'

        body = re.sub(r"\{IMAGE (\d+)\}", image, "\n".join(self.parts))
        # Medium splits a pasted code block at its first empty line, so give every empty line a single space
        body = re.sub(r"<pre>.*?</pre>", lambda match: re.sub(r"\n(?=\n)", "\n ", match.group(0)), body, flags=re.S)
        style = ("body { max-width: 720px; margin: 40px auto; font: 18px/1.6 Georgia, serif; color: #222; padding: 0 16px; }"
                 "pre { background: #f4f4f4; padding: 12px; overflow-x: auto; font-size: 14px; }"
                 "img { max-width: 100%; display: block; margin: 16px auto; }")
        return f'<!doctype html>\n<html><head><meta charset="utf-8"><style>{style}</style></head><body>\n{body}\n</body></html>\n'


def take_screenshots(notebook, screenshots: list[dict], render_path: Path, image_dir: Path):
    page_html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    page_html = page_html.replace("</head>", f"<style>{SCREENSHOT_CSS}</style></head>", 1)
    render_path.write_text(page_html)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_WIDTH, "height": 1200}, device_scale_factor=PIXEL_RATIO)
        page.goto(render_path.as_uri(), wait_until="networkidle")
        if page.evaluate("typeof MathJax !== 'undefined'"):
            page.evaluate("new Promise(resolve => MathJax.Hub.Queue(resolve))")  # wait until every equation is typeset
        else:
            print("warning: MathJax didn't load (no internet?), so equations will be screenshotted as raw LaTeX")

        selectors = {
            "cell": "",
            "output": ".jp-Cell-outputWrapper",
            "table": ".jp-RenderedMarkdown table",
            "equation": ".MJXc-display > .MathJax_CHTML",  # MathJax 2's CommonHTML output, as nbconvert configures it
        }
        for shot in screenshots:
            cell = page.locator(f'[id="cell-id={shot["cell_id"]}"]')
            element = cell.locator(selectors[shot["kind"]]).nth(shot["index"]) if selectors[shot["kind"]] else cell
            element.scroll_into_view_if_needed()
            box = element.bounding_box()
            if shot["kind"] == "output":  # crop to where the content ends, so short outputs don't come with empty space
                box["width"] = element.evaluate(CONTENT_WIDTH_JS)
            scroll_x, scroll_y = page.evaluate("[window.scrollX, window.scrollY]")
            page.screenshot(path=image_dir / shot["file"], full_page=True, clip={
                "x": max(box["x"] + scroll_x - PADDING, 0), "y": max(box["y"] + scroll_y - PADDING, 0),
                "width": box["width"] + 2 * PADDING, "height": box["height"] + 2 * PADDING,
            })
        browser.close()


def add_origin_note(article: Article, site_url: str, code_url: str):
    """Put a 'first published on my website' line straight after the title and subtitle, which open the first cell."""
    note = (f"<p><em>This article was first published on <a href=\"{html.escape(site_url)}\">my website</a>, where the code can be "
            f"copied and the tables and equations render properly. The full notebook, with every output shown here, is "
            f"<a href=\"{html.escape(code_url)}\">on GitHub</a>.</em></p>")
    first = article.parts[0]
    subtitle = re.match(r"\s*<h1>.*?</h1>\s*<p><em>.*?</em></p>", first, flags=re.S)
    if not subtitle:
        raise ValueError("expected the first cell to start with '# Title' and then '*Subtitle*'")
    article.parts[0] = first[: subtitle.end()] + "\n" + note + first[subtitle.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out", type=Path, help="output directory (default: outputs/medium/<notebook name>)")
    parser.add_argument("--keep-stderr", action="store_true", help="keep warnings: stderr output and vLLM's log lines (left out by default)")
    parser.add_argument("--site-url", help="the article's page on the website; adds a 'first published on my website' line under the subtitle")
    parser.add_argument("--code-url", help="where readers can get the notebook, used with --site-url (default: the notebook in GITHUB_REPO)")
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    args.code_url = args.code_url or f"{GITHUB_REPO}/blob/main/{notebook_path.relative_to(REPO).as_posix()}"
    out_dir = (args.out or REPO / "outputs" / "medium" / notebook_path.stem).resolve()
    image_dir = out_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)  # numbers shift when cells change, so start clean
    image_dir.mkdir(parents=True)

    notebook = nbformat.read(notebook_path, as_version=4)
    if not args.keep_stderr:  # warnings and progress bars are noise in an article
        log_line = re.compile(r"^(DEBUG|INFO|WARNING|ERROR) \d\d-\d\d \d\d:\d\d:\d\d \[")  # vLLM's logger, which prints to stdout
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            outputs = []
            for output in cell.outputs:
                if output.output_type == "stream":
                    if output.name == "stderr":
                        continue
                    output.text = "".join(line for line in output.text.splitlines(keepends=True) if not log_line.match(line))
                    if not output.text.strip():
                        continue
                outputs.append(output)
            cell.outputs = outputs
    article = Article(notebook_path.parent, image_dir)
    untagged, relative_links = [], []
    for cell_number, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            article.add_markdown(cell, cell_number)
            relative_links += [link for link in re.findall(r"\]\(([^)]+)\)", cell.source)
                               if not re.match(r"https?://|#|images/", link)]
        elif cell.cell_type == "code":
            tags = TAGS.intersection(cell.metadata.get("tags", []))
            if not tags:
                untagged.append(cell_number)
            article.add_code(cell, cell_number, tags.pop() if tags else "medium-screenshot")

    if args.site_url:
        add_origin_note(article, args.site_url, args.code_url)

    take_screenshots(notebook, article.screenshots, out_dir / "_render.html", image_dir)
    (out_dir / "article.html").write_text(article.render(with_images=False))
    (out_dir / "preview.html").write_text(article.render(with_images=True))

    if args.site_url:
        print(f"code link: {args.code_url}")
    print(f"{len(article.images)} images in {image_dir}")
    print(f"article: {out_dir / 'article.html'}   preview: {out_dir / 'preview.html'}")
    if untagged:
        print(f"untagged code cells (treated as medium-screenshot): {untagged}")
    if relative_links:
        print(f"relative links, which won't work on Medium: {sorted(set(relative_links))}")


if __name__ == "__main__":
    main()
