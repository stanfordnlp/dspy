# Updating lm15

Do not edit `lm15/` by hand. Make changes in the source repository and sync
`cmpnd-ai/lm15-python` first.

From the DSPy root, with all changes committed:

```sh
python scripts/update_vendored_lm15.py main
```

Pass a full source commit instead of `main` for a reproducible import.
The command fetches the cmpnd fork, splits its `lm15/` package history, and
uses `git subtree add` (first import) or `git subtree merge` (updates), with
squashed history. It creates local commits; it never pushes.

The first run replaces the old copied snapshot in a separate commit before
adding the subtree. Review the resulting diff before pushing. If a merge
conflicts, inspect Git's status and resolve or abort the merge before retrying;
the tool does not discard conflicts automatically.

`lm15-provenance.txt` records the source URL, Python commit, contract pin and
package split commit. `lm15-LICENSE` preserves the source repository's license.
Both live outside the subtree to keep its contents faithful to upstream.
The full upstream package is imported, including any non-Python files tracked
there. Packaging determines which files ship to users.

Merge subtree import/update PRs with a merge commit, not squash or rebase.
The subtree merge and squash-parent records must survive so the next update
can find the previous imported version.

The `Verify bundled lm15` workflow builds a source distribution, builds a wheel
from it, and tests a fresh installation outside the checkout. It checks public
exports, request/response conversion, pickle round trips, and license/provenance
files. No provider keys or network LM calls are needed.

`dspy.lm15` re-exports lm15's public top-level names; it does not alias lm15's
whole submodule tree. Use `from dspy.lm15 import Request`, not
`from dspy.lm15.types import Request`. Existing DSPy LM behavior is unchanged
by the vendor import.
