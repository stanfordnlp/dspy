---
name: writing-release-notes
description: Drafts DSPy release notes from maintainer-selected highlights and a verified, complete Git release range. Use when preparing or updating stable, beta, or release-candidate notes.
---

# Writing DSPy Release Notes

Ask what matters to the maintainer, then turn the complete release range into source-checked release notes in DSPy's established style.

## 1. Ask for highlights before drafting

Start by asking:

> Which PRs or features should headline this release? PR numbers, links, or feature names are fine. What version are we preparing, and is it stable, beta, or RC?

Use information already supplied rather than asking again. If highlights are missing, **wait for the maintainer's answer before drafting**. If they ask you to choose, inspect the range and propose highlights yourself. Their choices determine emphasis, not which changes are included in the full list.

Establish the previous release tag and target branch, tag, or commit. If unspecified, inspect releases and state a proposed baseline and target. Ask when multiple release lines make the scope ambiguous. Distinguish local `main` from `origin/main`; do not assume the checkout is current.

If the release label is undecided, explain briefly:

- Beta (`X.Y.Zb1`): APIs or behavior may still change based on testing and feedback.
- RC (`X.Y.Zrc1`): intended final behavior; would ship unchanged if no blockers are found.
- Both are prereleases. Do not describe either as stable.

For a first beta, normally use the preceding stable release as the baseline. For subsequent betas, RCs, and the final release, establish whether the maintainer wants cumulative notes since stable or incremental notes since the previous prerelease. Do not silently switch baselines.

## 2. Read precedents and establish the exact range

Read two or three published releases, including a comparable minor or patch release and a prerelease when relevant. Useful style references are [3.3.0](https://github.com/stanfordnlp/dspy/releases/tag/3.3.0), [3.3.1](https://github.com/stanfordnlp/dspy/releases/tag/3.3.1), and [3.2.0](https://github.com/stanfordnlp/dspy/releases/tag/3.2.0). Prefer newer comparable releases as conventions evolve.

Read `.github/.internal_dspyai/internals/release-checklist.md` for required release notices. This is a notes-writing task: report outstanding release prerequisites, but do not execute tagging, metadata refresh, publishing, or other release steps merely because the checklist mentions them.

Use Git and GitHub CLI, or equivalent read-only tools:

```bash
gh release list --repo stanfordnlp/dspy --limit 15
gh release view TAG --repo stanfordnlp/dspy --json body --jq .body
git status --short
git remote -v
git rev-parse --is-shallow-repository
# If shallow, fetch full history before inspecting the release range:
git fetch --quiet --unshallow origin
# Otherwise, or after unshallowing:
git fetch --quiet origin --tags
git rev-parse BASE_TAG TARGET_REF
git log --format='%H %s' BASE_TAG..TARGET_REF
git diff --stat BASE_TAG..TARGET_REF
```

Replace placeholders with the established refs. Record the resolved target commit as the draft cutoff. Do not reset or change the user's checkout to inspect it; use `git show TARGET_REF:path` and range diffs when the local files differ from the target.

Build an inventory of every change in the range, mapping commits to PR numbers, titles, URLs, and GitHub author logins:

```bash
gh pr view NUMBER --repo stanfordnlp/dspy --json number,title,body,author,url,mergeCommit
# For commits without a reliable PR number in the subject:
gh api repos/stanfordnlp/dspy/commits/SHA/pulls
```

Git reachability, not merge date or PR search results alone, determines inclusion. Deduplicate PRs spanning multiple commits. Account for direct commits explicitly. Investigate reverts, backports, and squash merges; do not advertise reverted functionality as shipped. Include maintenance, documentation, dependency, CI, and version-bump PRs, not just user-facing changes. Paginate API results if needed.

## 3. Verify what the release actually does

Read the highlighted PR bodies and relevant implementation, tests, and documentation at the target revision. Use PR titles as an index, not as sufficient evidence for behavioral claims. Resolve maintainer-provided feature names to actual changes in the range. Flag requested highlights that are not included rather than presenting them as shipped.

For each highlight, establish:

- What users can now do, or what failure is fixed.
- Defaults, opt-in switches, experimental status, and limits.
- Migration requirements: removed APIs versus deprecations with a future cutoff.
- Changes to return shapes, exceptions, dependencies, caching, concurrency, cost, or trust boundaries when relevant.
- Whether a previously published known issue is fixed, still present, or only partly addressed.

Do not imply async means parallel, a subprocess means a security sandbox, or a backend replacement means all legacy integrations are removed. Avoid performance or reliability claims unsupported by evidence. Attribute measurements and their conditions if included. Distinguish prior PR test reports from checks you actually run.

Use small examples checked against the target API. Do not invent credentials or make paid provider calls solely to draft notes. Syntax checks are not live-provider verification; state that limitation. Keep unresolved release-readiness concerns separate from the copy-ready release body.

## 4. Draft in DSPy's release style

Use this structure, omitting empty sections and adapting depth to the release:

```markdown
# DSPy VERSION

Short description of the release and the most consequential upgrade notice.

## Highlights

### Feature or outcome — @author

Explain the user benefit, important behavior, and limitations.
Include a short example when it makes the change easier to adopt.

PRs: linked PR numbers

## API and Compatibility Changes

Separate immediate breaks, future deprecations, dependency floors,
behavioral changes, and known limitations. Give concrete migration actions.

## Full PR List

### Category

- User-facing description by @author ([#NUMBER](https://github.com/stanfordnlp/dspy/pull/NUMBER)).

## Contributors

Thank human contributors; credit automation separately.

**Full Changelog:** https://github.com/stanfordnlp/dspy/compare/BASE_TAG...VERSION
```

Group the full list by the changes actually present: for example language models/adapters, agents/tools/interpreters, optimizers/evaluation, core APIs, documentation, and dependencies/CI/release engineering. Every PR appears exactly once **within the full list**; it may also be cited in highlights or migration notes. Grouped dependency bullets are acceptable only if each PR and its change remain identifiable. Include direct commits with commit links.

Use actual GitHub author logins, not inferred names or email addresses. Credit multiple authors where appropriate. Only add a first-time-contributors section if first contribution status has been verified against earlier repository history; absence from the preceding release is not proof.

For a beta or RC, add a prominent prerelease notice, an exact-version installation command, and focused requests for feedback. Keep the title, package version, and future comparison tag consistent. If the tag is not yet created, identify the comparison link as prospective in the handoff, not as a verified existing release.

Keep the prose focused on consequences and adoption, not implementation inventory. The full PR list supplies completeness; highlights supply emphasis. Match prior notes without copying stale limitations, version-specific promises, or unverified claims.

## 5. Audit completeness and hand off

Before delivering:

1. Compare the full-list PR IDs with the inventory from the exact Git range. Check for omissions, extras, and duplicates; account for direct commits and reverts.
2. Check author credits against GitHub and ensure the contributor list covers the credited human authors.
3. Verify version strings, baseline, comparison target, links, code fences, and example syntax. Distinguish lockfile updates from changed supported dependency ranges.
4. Re-read the draft for misleading defaults, missing migration actions, and claims stronger than the evidence.
5. Report the baseline, target cutoff, coverage count, and any verification limitations. Do not claim release readiness from a notes audit alone.

If the maintainer says another PR just merged, fetch again and inspect the delta from the recorded cutoff. Integrate the new behavior in both the relevant prose and full list, refresh credits, and rerun the completeness check. Do not change the original baseline. If only the version label changes, update the title, prerelease wording, installation command, filename when applicable, and comparison target together.

Return copy-ready Markdown in the conversation unless the maintainer requests a file or the established workflow already uses one. Preserve an existing draft's location and user edits when updating it. Do not add generated notes to the repository without a reason to track them.

Drafting notes does **not** authorize pushing, creating or publishing GitHub releases, tagging, changing package versions, or triggering release workflows. Perform those actions only with explicit authorization for the specific release operation.
