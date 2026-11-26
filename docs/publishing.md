# Publishing an article

Each article is a notebook in `tutorial/series/`. It is published in two places:

1. **The website** (https://chris-hughes10.github.io/, a Quarto site in the `Chris-hughes10.github.io` repo). This is
   the original.
2. **Medium.** A copy for distribution. Its canonical link points back to the website, and it links to the notebook
   on GitHub.

Both are built from the notebook's **saved outputs**. Nothing is re-run, so run the notebook first.

The order matters: GitHub repo public → website → Medium. The website links to the notebook on GitHub, and Medium's
canonical link needs the website URL.

## Per-article values

| | Article 1 |
|---|---|
| Notebook | `tutorial/series/grpo_manual.ipynb` |
| Slug | `grpo-from-scratch` |
| Website URL | https://chris-hughes10.github.io/posts/grpo-from-scratch/ |
| Code URL (the default) | https://github.com/Chris-hughes10/rl-from-scratch/blob/main/tutorial/series/grpo_manual.ipynb |
| Medium tags | Reinforcement Learning, LLM, Machine Learning, PyTorch, Deep Learning |

Commands below use article 1's values. For another article, substitute its row.

## 0. One-time setup

- **Quarto,** to preview the website locally (optional, see step 2). Install it from https://quarto.org/docs/get-started/.
  The site's CI uses the latest release.
- **Chromium for Playwright,** for the Medium export: `uv run --with playwright==1.49.1 playwright install chromium`.
- **The public GitHub repo.** The website and Medium articles link to it, so it must be public under the name in
  `GITHUB_REPO` (in `tools/notebook_to_quarto.py` and `tools/notebook_to_medium.py`), currently `rl-from-scratch`:
  1. On GitHub, rename the repo (Settings → General → Repository name). GitHub redirects the old URL.
  2. Locally, run `git remote set-url origin https://github.com/Chris-hughes10/rl-from-scratch.git`.
  3. Make it public (Settings → General → Danger Zone → Change visibility).
  4. Add a description and topics (e.g. `reinforcement-learning`, `llm`, `grpo`, `pytorch`, `verl`).
  5. Open the code URL above in a private window and check that the notebook renders.

## 1. Finish the notebook

- [ ] **Re-run it top to bottom,** so every output is fresh and has an execution count.
- [ ] **Check every number quoted in the prose against the new outputs.** Training results vary from run to run
      (bf16 non-determinism).
- [ ] **Remove references** to unpublished articles, or to anything else readers can't follow.
- [ ] **Tag every code cell** for the Medium export: `medium-code`, `medium-screenshot` or `medium-skip`. The rule,
      and a snippet that applies it, are in [tools/notebook_to_medium.md](../tools/notebook_to_medium.md) §1. The
      export lists any untagged cells.
- [ ] **Run `.venv/bin/python tools/check_series.py`,** so that `tutorial/lib/` still matches the notebook.
- [ ] **Commit and push this repo,** so the notebook on GitHub is the version the article describes.

## 2. Website

`tools/notebook_to_quarto.py` copies the notebook into the website repo as
`myblog/posts/<slug>/index.ipynb`. On the way, it:

- removes the title and subtitle lines, because Quarto shows them from the front matter
- removes warnings (stderr, and vLLM's log lines)
- adds a "Run it yourself" line that links to the notebook on GitHub
- copies the images the notebook uses

It writes `_metadata.yml` (the front matter) only if that file doesn't exist yet, so your edits there survive
re-syncing. It warns when the notebook's title or subtitle no longer matches `_metadata.yml`.

- [ ] **Sync:**

      python3 tools/notebook_to_quarto.py tutorial/series/grpo_manual.ipynb --slug grpo-from-scratch

  Re-run this after every edit to the notebook. It overwrites `index.ipynb` and the images.
- [ ] **Check `_metadata.yml`:**
  - the title and subtitle (fix any warning from the sync)
  - the `description`, used in listings and search
  - the `categories`
  - `reading-time`
- [ ] **Review the rendered post,** in one of two ways:
  - **Locally:** run `cd ../Chris-hughes10.github.io/myblog && quarto preview`, then open `/posts/<slug>/`. On the
    VM, open it through VS Code's port forwarding.
  - **On the real site, unlisted:** keep `draft: true`, then commit and push the website repo. The site uses
    `draft-mode: unlinked`, so the post is live at its URL but isn't listed, linked or in the sitemap. CI takes a
    few minutes.

  Look at the equations, the tables, the images and the "Run it yourself" link.
- [ ] **Publish:**
  1. In `_metadata.yml`, set `date` to today and delete the `draft: true` line.
  2. Commit and push the website repo.
  3. Check that the post appears under Technical Blog Posts and that its URL matches the table above.

## 3. Medium

- [ ] **Export:**

      uv run tools/notebook_to_medium.py tutorial/series/grpo_manual.ipynb \
          --site-url https://chris-hughes10.github.io/posts/grpo-from-scratch/

  It takes about 35 s and needs internet access, because MathJax loads from a CDN. It writes
  `outputs/medium/grpo_manual/`. `--site-url` adds an italic line under the subtitle: "first published on my
  website…", with a link to the notebook on GitHub. The output prints the code link it used, and lists any untagged
  cells and relative links. Details are in [tools/notebook_to_medium.md](../tools/notebook_to_medium.md).
- [ ] **Paste the rendered page, not the HTML source.** Copying the source pastes raw tags.
  1. Serve the folder: `cd outputs/medium/grpo_manual && python3 -m http.server 8765 --bind 127.0.0.1`.
  2. Open `http://localhost:8765/article.html` in your browser (through VS Code's port forwarding on the VM).
  3. Select all, copy, and paste into a new Medium story.
- [ ] **Add the images.** Download `images/` (in VS Code's explorer: right-click → Download…). Drag each image onto
      its `[IMAGE nn: …]` placeholder. Use `preview.html` to check what goes where.
- [ ] **Turn the title and subtitle into Medium's own.** The export pastes them as a heading and an italic line.
      Select each one and press T for the title, then the smaller t for the subtitle. Medium uses them in previews and
      search.
- [ ] **Check the links.** Click the "first published on my website" link and the GitHub link. Replace any relative
      links that the export listed with full URLs.
- [ ] **In the story's settings:**
  - Under **Advanced settings → Customize canonical link**, enter the website URL. Search engines then credit the
    website, not Medium.
  - Set the SEO title and description, e.g. "GRPO from scratch in PyTorch, checked against verl".
  - Add up to 5 tags (see the table above).
- [ ] **Publish.**

## 4. Afterwards

- [ ] Check that the README's article table links to the published post.
- [ ] If an earlier article mentions this one as upcoming, link it now, on the website and on Medium.
