# AWX/Ansible Log Viewer: A Keyboard-Driven TUI for Fast Incident Triage

This case study covers [AWX-Ansible-Log-Viewer](https://github.com/zombat/AWX-Ansible-Log-Viewer), an open-source terminal application for exploring large AWX/Ansible logs quickly.

Unlike a blog-style walkthrough, this page is written as project documentation: what problem it solves, how it is structured, why the UX decisions matter, and how to extend it.

## Project Snapshot

- **Repository:** [zombat/AWX-Ansible-Log-Viewer](https://github.com/zombat/AWX-Ansible-Log-Viewer)
- **Primary language:** Python
- **Interface:** Terminal UI powered by prompt_toolkit
- **Package:** `ansible-logviewer` on PyPI
- **License:** MIT

## Problem

Operational teams often need to inspect large AWX/Ansible logs during outages or deployment incidents. The default workflows are usually one of these:

- Scrolling raw logs in a terminal.
- Searching in a text editor with limited context.
- Writing one-off scripts for each investigation.

These approaches are slow when you need to repeatedly answer practical questions, such as:

- Which hosts failed?
- Which task failed first?
- Are failures clustered around one role or spread across many systems?

## Solution Overview

AWX-Ansible-Log-Viewer provides a focused TUI that prioritizes fast navigation and filtering:

- Multi-select filters for hosts, tasks, and statuses.
- Ordered status handling (`ok`, `changed`, `unreachable`, `failed`, `skipping`).
- Error navigation for failed/unreachable lines.
- Expand/collapse behavior for dense sections.
- Status and bottom bars for quick state awareness.

The result is less time spent on ad hoc parsing and faster root-cause discovery.

## Architecture at a Glance

The project separates concerns cleanly across modules:

- `ansible_logviewer/cli.py`: entry point and app wiring.
- `ansible_logviewer/log_manager.py`: parsing, indexing, filtering, navigation state.
- `ansible_logviewer/kb.py`: key bindings and modal/filter interactions.
- `ansible_logviewer/ui.py`: rendering styles and status/bottom bar presentation.
- `ansible_logviewer/lexer.py`: syntax highlighting for log lines.

This split helps keep parsing logic testable while allowing UI behavior to evolve without rewriting core filtering/navigation.

## UI Placeholder (to be replaced with real screenshot)

> Placeholder image slot:
> Add a full-width screenshot of the main TUI showing the log body, status bar, and filter state.
> Suggested asset path: `docs/docs/static/img/awx_ansible_log_viewer/main-ui.png`

## Interaction Model

### Keyboard-First Navigation

The app is intentionally keyboard-heavy to support fast incident triage:

- Movement and paging for long logs.
- Immediate jump to next/previous error clusters.
- Fast expand/collapse for noisy sections.
- Exit behavior with both safe and immediate options.

### Filtering Workflow

The filter dialog supports host/task/status filtering with multi-select controls, enabling iterative narrowing of high-volume logs without losing terminal flow.

## Getting Started

Install and run:

```bash
pip install ansible-logviewer
ansible-logviewer path/to/logfile.log
```

From source:

```bash
git clone https://github.com/zombat/AWX-Ansible-Log-Viewer.git
cd AWX-Ansible-Log-Viewer
python3 -m venv .venv
source .venv/bin/activate
pip install .
ansible-logviewer example.log
```

## CLI Options

Key options include:

- `--highlight-style underline|color|both`
- `--search-mode keyword|regex|both`
- `--debug`

These options make the viewer adaptable to different terminal capabilities and troubleshooting workflows.

## Why This Matters for Production Teams

The project shows a practical pattern for ops tooling:

- Keep the interface close to where engineers already work (the terminal).
- Model state explicitly (filters, collapsed sections, error index).
- Favor deterministic navigation and predictable key bindings over hidden gestures.

For environments where AWX/Ansible is central to deployments, this can reduce mean time to diagnosis during incidents.

## DSPy Extension Opportunities

This project is useful even without LLMs, but it is a strong foundation for DSPy-powered additions:

- **Structured failure summarization:** Generate concise incident reports from selected log spans.
- **Adaptive triage plans:** Use DSPy modules to propose next checks based on host/task/status patterns.
- **Retrieval over historical runs:** Combine parsed logs with retrieval modules to compare current failures against prior incidents.
- **Guardrailed explanations:** Produce human-readable root-cause hypotheses with explicit confidence and constraints.

A practical integration path is to keep the existing parser and navigation model unchanged, then add optional DSPy-backed commands for summary and diagnosis.

## Lessons from the Design

- Domain-specific filtering outperforms generic text search for repetitive incident work.
- Keyboard consistency and clear status feedback are critical for high-pressure debugging.
- Separating parse/index logic from UI rendering makes maintenance and testing significantly easier.

## Placeholder Diagram (to be replaced)

> Placeholder diagram slot:
> Add an architecture image showing data flow from raw AWX log -> parser/index -> filter state -> renderer.
> Suggested asset path: `docs/docs/static/img/awx_ansible_log_viewer/architecture.png`

## Links

- Repository: [AWX-Ansible-Log-Viewer](https://github.com/zombat/AWX-Ansible-Log-Viewer)
- Releases: [GitHub Releases](https://github.com/zombat/AWX-Ansible-Log-Viewer/releases)
- Package (if published): [PyPI: ansible-logviewer](https://pypi.org/project/ansible-logviewer/)
