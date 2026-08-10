# DSPy Vibe — VS Code extension

Turn a vibe-coding instruction into a structured brief, an `.agent`, and a
`.skill`, without leaving the editor. The extension is a thin client over the
[`dspy_vibe`](../dspy_vibe/README.md) Python package in this repository: it
collects the instruction and workspace context, runs the converter, and surfaces
the result where you are already looking.

## What it does

- **Vibe: Convert selection to brief** — select an instruction (a TODO, a loose
  paragraph) and get a brief in a side panel.
- **Vibe: Convert a typed instruction** — type the request into an input box.
- Open questions become editor **diagnostics**: unanswered ones are warnings,
  ones with an assumption are hints. This is the point of the tool — the gaps in
  a vibe instruction show up before you write code against a guess.
- **Vibe: Generate .agent and .skill from the last brief** — writes the
  artifacts into your configured directories. Existing files are never
  overwritten without a confirmation.
- **Vibe: Check the Python environment** — verifies the interpreter can run the
  converter, with an actionable message when it cannot.

Stack context is detected from your workspace manifests (`package.json`,
`pyproject.toml`, `Cargo.toml`, …) and passed to the converter, so the brief
does not ask "which stack?" when the answer is in the repo.

## Requirements

The converter is Python. Point the extension at an interpreter that has
`dspy_vibe` installed:

```bash
pip install -e .          # from the repository root
```

The extension resolves the interpreter in this order: the `dspyVibe.pythonPath`
setting, then the interpreter selected by the Microsoft Python extension, then
`python3`/`python` on `PATH`. Run **Vibe: Check the Python environment** if a
conversion fails.

Offline by default — with no `dspyVibe.model` set, the converter uses its
deterministic path and needs no API key. Set a model id (for example
`openai/gpt-4o-mini`) to use an LM.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `dspyVibe.pythonPath` | (auto) | Interpreter to run the converter. |
| `dspyVibe.model` | (empty) | LM id; empty runs the offline converter. |
| `dspyVibe.tools` | `Read, Edit, Bash` | Tools the generated agent may use. |
| `dspyVibe.agentDirectory` | `.claude/agents` | Where `.agent` files go. |
| `dspyVibe.skillDirectory` | `.claude/skills` | Where `.skill` files go. |
| `dspyVibe.specDirectory` | `docs/briefs` | Where brief documents go. |
| `dspyVibe.autoContext` | `true` | Detect the stack from manifests. |
| `dspyVibe.timeoutSeconds` | `120` | Converter timeout. |

## Develop

```bash
npm install
npm run compile
npm test          # unit tests; the integration test skips without dspy_vibe on PATH
```

Press F5 (**Run DSPy Vibe Extension**) to launch an Extension Development Host.

The design keeps `vscode` imports in `extension.ts` only; `runner.ts`,
`context.ts`, `artifacts.ts`, and `types.ts` are plain TypeScript and carry the
logic, which is why they can be tested with `node --test` alone.
