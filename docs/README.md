**If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**

&nbsp;

--------

&nbsp;

The content below is focused on how to modify the documentation site.

&nbsp;

# Modifying the DSPy Documentation

This website is rendered with [Zensical](https://zensical.org/). The repository's build script prepares the source files, renders the site, and generates the additional deployment artifacts used by dspy.ai.

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

3. (Optional) On macOS you may also need to install libraries for building the site:

   ```bash
   brew install cairo freetype libffi libjpeg libpng zlib
   export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
   ```

4. Run the same build entry point used by CI:

   ```bash
   python scripts/build_docs.py current --config mkdocs.yml --output site
   ```

This generates a static build in the `site` directory, including the API reference, rendered notebooks, redirects, social cards, sitemap, and `llms.txt`. Serve that directory to preview it locally:

```bash
python -m http.server --directory site 8000
```

If you see the build failing make sure to fix it before pushing.

## Continuous Integration (CI) Build Checks

We have automated build checks set up in our CI pipeline to ensure the documentation builds successfully before merging changes. These checks:

1. Run `python docs/scripts/build_docs.py current --config docs/mkdocs.yml --output docs/site`
2. Verify that the build completes without errors
3. Help catch potential issues early in the development process

If the CI build check fails, please review your changes and ensure the documentation builds correctly locally before pushing updates.

## Contributing to the `docs` Folder

This guide is for contributors looking to make changes to the documentation in the `dspy/docs` folder.

1. **Pull the up-to-date version of the website**: Please pull the latest version of the live documentation site via cloning the dspy repo. The current docs are in the `dspy/docs` folder.

2. **Push your new changes on a new branch**: Feel free to add or edit existing documentation and open a PR for your changes. Once your PR is reviewed and approved, the changes will be ready to merge into main.

3. **Updating the website**: Once your changes are merged to main, the changes would be reflected on live websites usually in 5-15 mins.

## LLMs.txt

The build process generates an `/llms.txt` file for LLM consumption, plus a Markdown copy of every rendered page at `<page>/index.md` for its links to point to. Configure sections in `build.yml` under `llms`; the generator lives in `scripts/zensical_build.py`.
