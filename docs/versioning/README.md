# Versioned documentation

The documentation site is published as one static site managed by Zensical's
[Mike fork](https://github.com/squidfunk/mike):

- `/` redirects to `/current/`, which is built from `main` and replaced on each
  successful Current deployment.
- `/X.Y.Z/` is an immutable snapshot built from tag `X.Y.Z` while importing
  the exact released DSPy wheel. `/X.Y/` redirects to the newest imported
  patch in that minor line.

Mike owns the generated `versioned-docs` branch, `versions.json`, default
redirect, minor aliases, and the selector rendered by Material or Zensical.
The official fork is pinned to an exact commit because it is currently
distributed from GitHub rather than PyPI. `releases.json` remains the source of
truth only for the one-time historical bootstrap. Prereleases are never
published as stable documentation routes.

The generated branch is the deployment output: configure the existing static
host to deploy `versioned-docs` instead of rebuilding the source branch. Mike
stores every release in that one branch; it does not create one hosting
deployment or project per version.

## Historical fidelity

Versions 3.0 through 3.3 use their tagged Material for MkDocs configuration.
Their requirements were not fully pinned and referred to DSPy on the moving
`main` branch. Bootstrap replaces that dependency with the release wheel and
resolves the remaining requirements no later than the tag's commit time.

Every release tag contains `uv.lock`, but those locks describe DSPy's project
and development dependencies. The separate documentation toolchain is defined
only in `docs/requirements.txt`; MkDocs, Material, Jupyter, redirects,
mkdocstrings, and llmstxt are absent from the locks. The locks therefore cannot
reproduce the historical documentation environment.

Each snapshot contains `_meta/build.json` with:

- source tag and commit;
- renderer and renderer version;
- DSPy artifact source and SHA-256;
- the complete resolved Python package set;
- known renderer differences.

PyPI never published `dspy==3.1.1`. That snapshot is the sole exception: its
wheel is built from tag `3.1.1` using the same metadata substitutions as the
release workflow and is marked `tag-built-wheel`.

Generated HTML is expected to differ byte-for-byte from the historical
deployment because the original build environment and output were not
archived. The compatibility contract is URLs, redirects, anchors, content,
notebooks, API symbols from the matching artifact, search behavior, social
metadata, navigation, and user-facing interactions. Visual review catches
accidental regressions, but exact fonts, spacing, line wrapping, card styling,
and pixel output are not part of the contract. Mike's version selector, build
metadata, HTML minification, and omission of production source maps are
intentional additions.

## Output size

Production builds use Mike's redirect aliases so `/X.Y/` contains redirect
HTML rather than another asset copy. They also remove source maps and minify
HTML with whitespace-sensitive elements preserved. Both renderers pass the
same static and browser parity gates after this optimization. Git deduplicates
byte-identical files in the generated branch; no content-addressed asset
rewriter or separate snapshot store is introduced.

## Zensical parity gate

Current and future snapshots may change from `material` to `zensical` only
after the parity checks cover every row below. Record any accepted difference
in the snapshot metadata rather than allowing silent omissions.

| Capability | Material pipeline | Zensical 0.0.57 status |
| --- | --- | --- |
| Markdown, navigation, CSS and JavaScript | Native | Native or Zensical compatibility support; feature checks pass |
| Template overrides | Jinja | MiniJinja compatibility; the incompatible custom tabs partial is replaced by Zensical's built-in tabs |
| API reference | `mkdocstrings` | Same plugin and matching public symbols |
| Notebooks | `mkdocs-jupyter` | Pre-render with `nbconvert` into a temporary source tree |
| Redirects | `mkdocs-redirects` | Generate equivalent static redirects after rendering |
| Social cards and metadata | Material `social` | Generate cards and inject equivalent metadata after rendering |
| `llms.txt` | `mkdocs-llmstxt` | Generate from the same source inventory after rendering |
| Build-time statistics | MkDocs hook | Run the existing fetcher before rendering and inject `extra.stats` |
| Search | Material search | Disco; indexed routes and representative browser discoverability pass, while result order may differ |
| Version deployment | Mike branch and selector | Official Zensical Mike fork using the same branch contract |

Update this table and the parity fixtures whenever Zensical or a replacement
extension changes one of these outcomes.

The executable gate is split into two layers:

```bash
python docs/scripts/zensical_parity.py \
  --material-output site-material \
  --zensical-output site-zensical \
  --report zensical-parity.json
python docs/scripts/zensical_browser_parity.py \
  --material site-material \
  --zensical site-zensical \
  --report zensical-browser-parity.json \
  --screenshots zensical-parity-screenshots
```

Both layers use the same `build_zensical_site` implementation as the production
entry point. The static layer fails closed on route, content, API, notebook,
redirect, metadata, social-card availability, `llms.txt`, search-index, asset,
template, statistics, sitemap, navigation, and version-selector regressions.
The browser layer fails closed on Mike's picker, search discoverability, desktop
and mobile navigation, dark-mode persistence, announcements, homepage and
tutorial interactions, footer navigation, and custom scripts. Screenshot and
social-card pixel comparisons are retained as informational review artifacts.

A fresh local run passes both layers. The `Zensical docs parity` workflow runs
on affected pull requests and manual dispatches, uploads reports and
screenshots even when a gate fails, and then enforces both reports. Current and
future release workflows explicitly select `zensical`. The release workflow
builds only the new tag, gives its minor alias to the newest patch, and pushes
the updated generated branch. Historical snapshots remain immutable and are
never rebuilt during or after the cutover.
