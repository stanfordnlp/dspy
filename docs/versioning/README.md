# Versioned documentation

DSPy's versioned documentation is one static site managed by the Zensical
team's [Mike fork](https://github.com/squidfunk/mike):

- `/` redirects to `/current/`.
- `/current/` is the mutable documentation built from `main`.
- `/X.Y.Z/` is a release snapshot built from tag `X.Y.Z` while importing the
  exact released DSPy wheel.
- `/X.Y/` redirects to the newest imported patch in that minor line.

The picker lists Current and every patch release. Minor aliases are navigation
conveniences and are hidden from the picker. Mike owns `versions.json`, the
default redirect, aliases, and version directories on the generated
`versioned-docs` branch in `krypticmouse/dspy-docs`.

Historical snapshots continue to use Material for MkDocs. Current uses
Zensical after passing the parity gate below. Stored static versions do not
need to share a renderer.

## Staging and cutover

The existing deployment repository and Vercel project remain in place. The
bootstrap and Current workflows assemble the complete candidate site on the
non-production `versioned-docs` branch. They do not change the live `master`
branch.

After the generated tree is reviewed, open a pull request in
`krypticmouse/dspy-docs` that applies that exact tree to `master`. Create a
backup branch before merging that deployment pull request. Vercel continues to
serve `master`, so no domain, project, or branch-setting migration is required.

Existing unversioned page URLs remain valid. Publishing Current generates root
redirect pages such as `/api/` → `/current/api/`, and each build scopes
hand-authored root-relative links to its own version so an old page cannot
silently jump into Current. Query strings and fragments survive redirects.

## Production publication and rollback

The initial versioned Material site and the later Zensical Current update are
reviewed as pull requests from `versioned-docs` to `master` in
`krypticmouse/dspy-docs`. Before merging the initial deployment pull request,
preserve the existing site once:

```bash
git fetch origin master versioned-docs
test -z "$(git ls-remote --heads origin legacy-material-backup)"
git push origin origin/master:refs/heads/legacy-material-backup
```

The command intentionally fails if the backup already exists, so a later
operation cannot silently replace the rollback point. Merge the reviewed
deployment pull request normally. To roll back the complete migration, restore
the backup tree with a normal commit:

```bash
git fetch origin master legacy-material-backup
git switch -C rollback-material-docs origin/master
git read-tree --reset -u origin/legacy-material-backup
git commit -m "Restore Material documentation"
git push origin HEAD:master
```

Current and release publishing fail closed unless `master` already contains
Mike's `versions.json`. Production automation is enabled only after both
deployment pull requests have been reviewed and merged.

## Historical fidelity

Versions 3.0 through 3.3 use the documentation source and Material
configuration from their release tags. Their requirements were not fully
pinned and referred to DSPy on the moving `main` branch. Bootstrap replaces
that dependency with the release wheel and resolves the remaining requirements
no later than the tag's commit time.

The tags' `uv.lock` files describe DSPy's project and development dependencies,
not the separate toolchain in `docs/requirements.txt`. They do not lock MkDocs,
Material, Jupyter, redirects, mkdocstrings, or llmstxt.

Each snapshot contains `_meta/build.json` with its source tag and commit,
renderer version, DSPy artifact source and SHA-256, complete resolved Python
package set, and known reconstruction differences.

PyPI never published `dspy==3.1.1`. That snapshot is the sole exception: its
wheel is built from tag `3.1.1` using the release workflow's metadata
substitutions and is marked `tag-built-wheel`.

The original generated deployments were not archived, so reconstructed HTML
is not expected to be byte-identical. The compatibility contract is routes,
redirects, anchors, content, notebooks, matching API symbols, search behavior,
metadata, navigation, and user-facing interactions. The version selector,
build metadata, conservative HTML minification, and omitted source maps are
intentional additions.

## Historical corrections

Release snapshots are immutable to automation, not permanent write-once
storage. An identical retry is a no-op; a retry with different output fails
before Mike can replace `/X.Y.Z/`.

An intentional correction is a reviewed pull request directly against
`krypticmouse/dspy-docs`, normally limited to the affected version directory.
That repository's pull request and Git history provide audit and rollback.

## Output size

Minor aliases contain redirects rather than duplicate assets. Production
builds remove source maps and conservatively minify HTML while preserving
whitespace-sensitive elements. Git deduplicates byte-identical objects in the
deployment repository. Browsers request only the selected page and its assets;
they do not download the aggregate repository.

## Zensical parity gate

Zensical does not directly run every plugin from the Material pipeline. The
production builder preserves their outputs at explicit compatibility
boundaries:

| Existing feature | Zensical path |
| --- | --- |
| API reference | the same `mkdocstrings` configuration and public symbols |
| Notebooks | pre-render with `nbconvert` in a disposable source tree |
| Redirects | emit equivalent static redirects after rendering |
| Social cards | generate per-page cards and inject matching metadata |
| `llms.txt` | generate from the same configured source inventory |
| Build-time statistics | run the existing fetcher before rendering |
| Search | use Disco and require route coverage plus representative discoverability |
| Custom tabs override | use Zensical's built-in tabs implementation |

The static parity gate compares all generated routes, article headings and
content, API symbols, notebooks, redirects, metadata, social-card availability,
`llms.txt`, search inventory, assets, sitemap, navigation, and version-picker
inventory. The browser gate exercises desktop and mobile navigation, search,
dark-mode persistence, announcements, footer links, homepage and tutorial
interactions, custom scripts, and picker behavior.

Screenshots and pixel-difference measurements are review artifacts rather than
pass/fail criteria. Zensical may differ in typography, spacing, wrapping,
navigation fitting, code rendering, search ranking, and social-card appearance
without dropping a feature.
