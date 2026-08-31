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

This first migration stage continues to render Current and all historical
snapshots with Material for MkDocs. Changing Current's renderer is a separate
step; stored static versions do not need to share a renderer.

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
