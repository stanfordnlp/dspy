# Contributing to DSPy

Thank you for your interest in contributing to DSPy! We appreciate contributions of all kinds, including bug fixes, feature additions, documentation improvements, and more. This guide will help you get started with the contribution process.

## Table of Contents


1. [Development Environment Setup](#development-environment-setup)
   - [Option 1: VSCode Dev Container (Recommended)](#option-1-vscode-dev-container-recommended)
   - [Option 2: Manual Setup](#option-2-manual-setup)
2. [Development Workflow](#development-workflow)
3. [Testing](#testing)
4. [Pre-commit Hooks](#pre-commit-hooks)
5. [Contributing Process](#contributing-process)
6. [Code Style and Documentation](#code-style-and-documentation)
7. [Commit Message Format](#commit-message-format)
8. [Getting Help](#getting-help)



## Development Environment Setup

### Option 1: VSCode Dev Container (Recommended)

Using VSCode dev containers provides a pre-configured development environment with all necessary requirements, IDE extensions, and settings.

1. Prerequisites:
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Install [VSCode](https://code.visualstudio.com/)
   - Install the [Remote Development extension](ms-vscode-remote.vscode-remote-extensionpack) in VSCode

2. Setup Steps:
   1. Open the VSCode command palette (Cmd/Ctrl + Shift + P)
   2. Select "Dev Containers: Rebuild and Reopen in container"
   3. Wait for the container to build and initialize
   4. Select the Poetry interpreter when prompted (check build logs for instructions)
   5. Verify setup by running `pytest` in the terminal

Note: After initial setup, you can re-enter the container using "Dev Containers: Reopen in container" without rebuilding.

### Option 2: Manual Setup

1. Fork and Clone the Repository:
```bash
git clone https://github.com/YOUR-USERNAME/dspy.git
cd dspy
```

2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install Dependencies:
```bash
poetry install --with dev
```

4. Activate the Virtual Environment:
```bash
poetry shell
```

## Development Workflow

1. Create a New Branch:
```bash
git checkout -b your-branch-name
```

2. Make Your Changes
3. Run Tests:
```bash
pytest
```

4. Run Pre-commit Checks:
```bash
pre-commit run --all-files
```

## Testing

Run all tests:
```bash
pytest
```

Run specific test suite:
```bash
pytest tests/PATH_TO_TEST_SUITE
```

Testing CI Actions Locally:
- Install [act](https://nektosact.com/introduction.html)
- Run: `act push` (use `--container-architecture linux/amd64` flag for M1/M2 Macs)

## Pre-commit Hooks

1. Install the hooks:
```bash
pre-commit install
```

2. Run manually when needed:
```bash
pre-commit run --all-files
```

## Contributing Process

1. Create an Issue
   - Check the [issue tracker](https://github.com/stanfordnlp/dspy/issues) first
   - [Create a new issue](https://github.com/stanfordnlp/dspy/issues/new) if needed

2. Submit a Pull Request:
   - Push your changes: `git push origin your-branch-name`
   - Open a PR on GitHub
   - Link the issue: "Closes #ISSUE_NUMBER"
   - Provide a clear description of changes

## Code Style and Documentation

1. Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines
2. Run style checks: `flake8`
3. Write comprehensive docstrings:
```python
def example_function(param1, param2):
    """
    This function processes input and returns a boolean.

    Args:
        param1 (str): The first parameter.
        param2 (int): The second parameter.

    Returns:
        bool: Returns True if successful, False otherwise.
    """
    return True
```

## Commit Message Format

Commits must follow this format:
```
type(scope): description
```

Where:
- `type` is one of: break, build, ci, docs, feat, fix, perf, refactor, style, test, ops, hotfix, release, maint, init, enh, revert
- `scope` is the area of change (e.g., dspy, devcontainer)
- `description` is a brief explanation of the change

Example:
```
feat(dspy): add new feature
enh(devcontainer): decrease image size
```

## Getting Help

- Join the discussion in the repository's Discussions section
- Open an issue for specific problems or questions
- Review existing documentation and examples in the `examples/` directory

We look forward to your contributions!