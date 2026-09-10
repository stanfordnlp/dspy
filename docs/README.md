**If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**

&nbsp;

--------

&nbsp;

The content below is focused on how to modify the documentation site.

&nbsp;

# Modifying the DSPy Documentation


This website is built using [Material for MKDocs](https://squidfunk.github.io/mkdocs-material/), a Material UI inspired theme for MKDocs.

## Building docs locally

To build and test the documentation locally:

1. Navigate to the `docs` directory:
   ```bash
   cd docs
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. In docs/ directory, run the command below to generate the API docs and index them:
   ```bash
   python scripts/generate_api_docs.py
   python scripts/generate_api_summary.py
   ```

4. (Optional) On MacOS you may also need to install libraries for building the site
   ```bash
   brew install cairo freetype libffi libjpeg libpng zlib
   export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
   ```

5. Run the build command:
   ```bash
   mkdocs build
   ```

This will generate a static build of the documentation site in the `site` directory. You can then serve this directory to view the site locally using:

```bash
mkdocs serve
```

If you see the build failing make sure to fix it before pushing.

## Continuous Integration (CI) Build Checks

We have automated build checks set up in our CI pipeline to ensure the documentation builds successfully before merging changes. These checks:

1. Run the `mkdocs build` command
2. Verify that the build completes without errors
3. Help catch potential issues early in the development process

If the CI build check fails, please review your changes and ensure the documentation builds correctly locally before pushing updates.

## Contributing to the `docs` Folder

This guide is for contributors looking to make changes to the documentation in the `dspy/docs` folder. 

1. **Pull the up-to-date version of the website**: Please pull the latest version of the live documentation site via cloning the dspy repo.  The current docs are in the `dspy/docs` folder.

2. **Push your new changes on a new branch**: Feel free to add or edit existing documentation and open a PR for your changes. Once your PR is reviewed and approved, the changes will be ready to merge into main. 

3. **Updating the website**: Once your changes are merged to main, the changes would be reflected on live websites usually in 5-15 mins.

## LLMs.txt

The build process generates an `/llms.txt` file for LLM consumption using [mkdocs-llmstxt](https://github.com/pawamoy/mkdocs-llmstxt). Configure sections in `mkdocs.yml` under the `llmstxt` plugin. The plugin also writes every page as `<page>/index.md`, which is what the agentic search below indexes.

## "Ask AI" (toast-1)

The published pages are indexed in a Mixedbread store (`dspy-docs`) and the
Python sources under `dspy/` in `dspy-code`. `scripts/sync_search_index.py`
keeps both stores in step with the repo (incremental, sha256-keyed); CI runs
it after `mkdocs build` on pushes to `main` that touch `docs/` or `dspy/`
(`.github/workflows/docs-search-sync.yml`), using the `MXBAI_API_KEY`
repository secret. To re-index by hand:

```bash
pip install mixedbread
python3 docs/scripts/sync_search_index.py --site docs/site   # after `mkdocs build`
```

`docs/js/toast-chat.js` adds the "Ask AI" panel to every page. Its backend is
the Vercel Edge Function in `api/chat.js`, which ships with the site (this
directory is mirrored into the Vercel project) and runs on the same origin.
Set `MXBAI_API_KEY` and `OPENROUTER_API_KEY` in the Vercel project's
environment variables and it is live; the browser never sees the keys. By
default it runs the pipeline that won the docs-harness A/B (toast-1 agentic
search, whole-page reading of the top results, an answerer model); `MODE=toast`
lets toast-1 answer on its own with `api/system_prompt.txt`.

To preview the chat locally, put the two keys in `docs/api/.env.local`
(gitignored) and run the dev server next to `mkdocs serve`:

```bash
node docs/api/dev.mjs        # serves the function on http://localhost:8787
```

To route the widget at an external proxy instead, set
`extra.toast_chat_endpoint` in `mkdocs.yml`.

