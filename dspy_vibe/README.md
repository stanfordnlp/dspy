# dspy_vibe

Turn vibe-coding instructions into artifacts a coding agent can actually
execute: a structured brief, an `.agent` definition, and a reusable `.skill`.

```
"csinálj egy login formot, legyen szép és gyors, ne kelljen backend"
        │
        ▼
   VibeSpec ──────┬──────────────┐
 goal, scope,     │              │
 non-goals,       ▼              ▼
 constraints,  .agent         .skill
 acceptance,   who does it    the reusable method
 risks,        + guardrails   + triggers, checks, limits
 open questions
```

The point is the middle box. A vibe instruction is underspecified on purpose,
and handing it straight to an agent means the agent silently invents the missing
half. The brief makes those gaps visible as open questions with the assumption
applied, so you can correct them before any code is written.

## Install

The package ships with this repository. From the repo root:

```bash
pip install -e .
```

## Use it

Offline — no API key, no network, deterministic:

```bash
python -m dspy_vibe convert "csinálj egy login formot, ne kelljen backend" --out ./vibe-out
```

```
spec   vibe-out/csinalj-egy-login-formot.spec.md
agent  vibe-out/csinalj-egy-login-formot.agent
skill  vibe-out/csinalj-egy-login-formot.skill

bundle quality: 0.89  (confidence: LOW)
```

With an LM:

```bash
python -m dspy_vibe convert "add a dark mode toggle" --model openai/gpt-4o-mini --out ./vibe-out
```

Useful flags: `--format json` for machine-readable artifacts, `--tools` to limit
the generated agent to tools your host actually offers, `--context` to pass
stack facts, `--stdout` to skip writing files, `--overwrite` to replace existing
artifacts (refused by default).

From Python:

```python
import dspy
from dspy_vibe import VibeCoder, write_bundle

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
result = VibeCoder(available_tools=["Read", "Edit", "Bash"])(
    instruction="add a dark mode toggle to the settings page",
    repo_context="Next.js app router, Tailwind, pnpm",
)
write_bundle(result.bundle, "./vibe-out")
```

Without a configured LM every module falls back to the deterministic converter,
so the same code runs in CI.

## Optimize it

`VibeCoder` is a plain `dspy.Module`, so any DSPy optimizer works on it. The
metrics in `metrics.py` are mechanical — no LM judge to talk into a good score:

| Metric | What it measures |
| --- | --- |
| `spec_faithfulness` | Fraction of scope claims anchored in the instruction. Catches invented work. |
| `spec_completeness` | Are goal, scope, acceptance, and verification present. |
| `spec_specificity` | Did it decompose the request or just echo it back. |
| `spec_honesty` | Are undecided points surfaced instead of silently decided. |
| `agent_validity` / `skill_validity` | Is the artifact executable and bounded. |

`spec_quality` combines the first four, with faithfulness as a *multiplier*
rather than a term: copying the instruction verbatim scores perfectly on
faithfulness, so as a bonus it would reward the laziest possible brief.

The deterministic baseline deliberately does not score full marks (~0.78 on
`spec_quality` over the example dev set), which is what leaves an optimizer
something to improve. See `examples/optimize.py`.

## Artifact formats

Both emitters write the same validated model.

`markdown` (default) — YAML frontmatter plus a Markdown body, the shape agent
hosts read directly:

```markdown
---
name: dark-mode-agent
description: Adds a dark mode toggle to the settings page.
tools:
  - Edit
  - Read
source_spec: dark-mode-toggle
---

# dark-mode-agent
...
## Guardrails
- Out of scope: changes to the login screen
```

`json` — the same fields as data, for programmatic pipelines. Validate them
later with `python -m dspy_vibe check out/*.agent`.

## What it does not do

- It does not write code. It produces the brief and the definitions; the agent
  you hand them to does the work.
- It does not decide your open questions. Blocking questions are reported and
  the exit output lists them.
- The offline converter is keyword heuristics, and labels itself `LOW`
  confidence for that reason. It is a floor, not a substitute for a model.
