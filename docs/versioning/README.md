# Versioned documentation

DSPy's versioned documentation is one static site managed by
[Mike](https://github.com/jimporter/mike):

- `/` redirects to `/current/`.
- `/current/` is the mutable documentation built from `main`.
- `/X.Y.Z/` is a release snapshot built from tag `X.Y.Z` while importing the
  exact released DSPy wheel.
- `/X.Y/` redirects to the newest imported patch in that minor line.

The picker lists Current and every patch release. Minor aliases are navigation
conveniences and are hidden from the picker. Mike owns `versions.json`, the
default redirect, aliases, and version directories on the generated
`master` branch in `krypticmouse/dspy-docs`.

Historical snapshots use Material for MkDocs, while Current and future release
snapshots use Zensical. Stored static versions do not need to share a renderer.

## Deployment

Current and release publication update production `master` in
`krypticmouse/dspy-docs`. Both paths require Mike metadata identifying Current
as Zensical before they write, so an unversioned or unexpected deployment fails
closed. Corrections use reviewed pull requests in that repository.

Existing unversioned page URLs remain valid. Publishing Current generates root
redirect pages such as `/api/` → `/current/api/`, and each build scopes
hand-authored root-relative links to its own version so an old page cannot
silently jump into Current. Query strings and fragments survive redirects.

## Production publication

Current publication follows the renderer state described above. Release
publication fails closed unless production's Mike metadata identifies Zensical
as the reviewed Current renderer.

After a stable `dspy` wheel reaches PyPI, the release workflow preserves that
exact wheel, builds `/X.Y.Z/` from the tag, and publishes it through Mike after
the existing package release job succeeds. Release-tag jobs never use GitHub's
lossy pending-concurrency slot. Mutable Current keeps latest-wins serialization
because a newer `main` build includes the superseded commit. Deployment writes
retry optimistic Git pushes; every release rechecks the Zensical promotion
marker after refetching, and a delayed older patch cannot move an `/X.Y/` alias
backward.

Corrections and rollbacks use reviewed pull requests in the deployment
repository. Restore a known-good tree with a new commit rather than rewriting
production history.

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
build metadata, and omitted source maps are intentional additions.

## Historical corrections

Release snapshots are immutable to automation, not permanent write-once
storage. An identical retry is a no-op; a retry with different output fails
before Mike can replace `/X.Y.Z/`.

An intentional correction is a reviewed pull request directly against
`krypticmouse/dspy-docs`, normally limited to the affected version directory.
That repository's pull request and Git history provide audit and rollback.

## Output size

Minor aliases contain redirects rather than duplicate assets. Production
builds remove source maps. Git deduplicates byte-identical objects in the
deployment repository. Browsers request only the selected page and its assets;
they do not download the aggregate repository.

## Zensical feature ownership

Zensical owns rendering, navigation, search, minification, API reference, and
sitemap generation through its native configuration. The small compatibility
builder owns only features Zensical does not provide:

| Existing feature | Implementation |
| --- | --- |
| Notebooks | pre-render with `nbconvert` in a disposable source tree |
| Redirects | emit static redirects from the configured route map |
| Social cards | generate per-page cards and inject matching metadata |
| `llms.txt` | generate from the same configured source inventory |
| Build-time statistics | run the existing fetcher into the disposable config |
| Compressed sitemap | gzip Zensical's generated `sitemap.xml` |

Redirect sources that overlap legacy pages are removed only from the disposable
tree before the static redirects are emitted. Source files in the repository
are unchanged.

Unit tests exercise each compatibility transform directly. The documentation
pull-request job then builds the complete site, making the native Zensical
configuration and all compatibility steps executable review evidence. Visual
review covers representative desktop and mobile pages; typography, spacing,
wrapping, search ranking, and card appearance may change without dropping a
feature.
