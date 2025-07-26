# Contribution Guide

DSPy is an actively growing project and community! We welcome your contributions and involvement. This guide will walk you through the entire contribution process, from finding issues to getting your code merged.

## Table of Contents

- [Finding an Issue](#finding-an-issue)
- [Contributing Code](#contributing-code)
- [Environment Setup](#environment-setup)
- [Testing](#testing)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

## Finding an Issue

The fastest way to contribute is to find open issues that need an assignee. We maintain two lists of GitHub tags for contributors:

- [good first issue](https://github.com/stanfordnlp/dspy/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22):
  a list of small, well-defined issues for newcomers to the project.
- [help wanted](https://github.com/stanfordnlp/dspy/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22):
  a list of issues that welcome community contributions. These issues have a wide range of complexity.

We also welcome new ideas! If you would like to propose a new feature, please open a feature request to
discuss. If you already have a design in mind, please include a notebook/code example to demonstrate
your idea. Keep in mind that designing a new feature or use case may take longer than contributing to
an open issue.

## Contributing Code

Follow these steps to submit your code contribution.

### Step 1. Open an Issue

Before making any changes, we recommend opening an issue (if one doesn't already exist) and discussing your
proposed changes. This way, we can give you feedback and validate the proposed changes.

If your code change involves fixing a bug, please include a code snippet or notebook
to show how to reproduce the broken behavior.

For minor changes (simple bug fixes or documentation fixes), feel free to open a PR without discussion.

### Step 2. Make Code Changes

To make code changes, fork the repository and set up your local development environment following the
instructions in the "Environment Setup" section below.

### Step 3. Commit Your Code and Run Autoformatting

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and use `ruff` for both linting and formatting. To ensure consistent code quality, we use pre-commit hooks that automatically check and fix common issues.


First you need to set up the pre-commit hooks (do this once after cloning the repository):

```shell
pre-commit install
```

Then stage and commit your changes. When you run `git commit`, the pre-commit hook will be
automatically run.

```shell
git add .
git commit -m "your commit message"
```

If the hooks make any changes, you'll need to stage and commit those changes as well.

You can also run the hooks manually:

- Check staged files only:

  ```shell
  pre-commit run
  ```

- Check specific files:

  ```shell
  pre-commit run --files path/to/file1.py path/to/file2.py
  ```

Please ensure all pre-commit checks pass before creating your pull request. If you're unsure about any
formatting issues, feel free to commit your changes and let the pre-commit hooks fix them automatically.

### Step 4. Create a Pull Request

Once your changes are ready, open a pull request from your branch in your fork to the main branch in the
[DSPy repo](https://github.com/stanfordnlp/dspy).

### Step 5. Code Review

Once your PR is up and passes all CI tests, we will assign reviewers to review the code. There may be
several rounds of comments and code changes before the pull request gets approved by the reviewer.

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.

## Environment Setup

Python 3.10 or later is required.

Setting up your DSPy development environment requires you to fork the DSPy repository and clone it locally.
If you are not familiar with the GitHub fork process, please refer to [Fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). After creating the fork, clone
it to your local development device:

```shell
git clone {url-to-your-fork}
cd dspy
```

Next, we must set up a Python environment with the correct dependencies. There are two recommended ways to set up the
dev environment.

### [Recommended] Set Up Environment Using uv

[uv](https://github.com/astral-sh/uv) is a rust-based Python package and project manager that provides a fast
way to set up the development environment. First, install uv by following the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).

After uv is installed, in your working directory (`dspy/`), run:

```shell
uv sync --extra dev
```

Then you are all set!

To verify that your environment is set up successfully, run some unit tests:

```shell
uv run pytest tests/predict
```

Note: You need to use the `uv run` prefix for every Python command, as uv creates a Python virtual
environment and `uv run` points the command to that environment. For example, to execute a Python script you will need
`uv run python script.py`.

### Set Up Environment Using conda + pip

You can also set up the virtual environment via conda + pip, which takes a few extra steps but offers more flexibility. Before starting,
make sure you have conda installed. If not, please follow the instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

To set up the environment, run:

```shell
conda create -n dspy-dev python=3.11
conda activate dspy-dev
pip install -e ".[dev]"
```

Then verify the installation by running some unit tests:

```shell
pytest tests/predict
```

## Testing

We strongly encourage adding tests for any new functionality or bug fixes. DSPy uses pytest for testing.

### Running Tests

To run the full test suite:

```shell
# Using uv
uv run pytest

# Using conda/pip
pytest
```

To run specific tests:

```shell
# Test a specific module
uv run pytest tests/predict

# Test a specific file
uv run pytest tests/test_example.py

# Test with coverage
uv run pytest --cov=dspy tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention `test_*.py` for test files
- Use descriptive test function names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies when appropriate

## Documentation

Good documentation helps users understand and adopt DSPy. When contributing:

### Code Documentation

- Add docstrings to all public functions, classes, and modules
- Follow Google-style docstrings
- Include parameter types, return types, and examples where helpful

### Examples and Tutorials

- Update existing examples if your changes affect them
- Consider adding new examples for significant new features
- Ensure examples are self-contained and well-commented

### README Updates

- Update the main README.md if your contribution adds new features
- Keep installation instructions current
- Update the feature list if applicable

## Getting Help

If you need help at any point during the contribution process:

- **Questions about the code**: Open a discussion on GitHub
- **Technical issues**: Create an issue with detailed information
- **Feature discussions**: Use GitHub discussions for broader conversations
- **Stuck on something**: Don't hesitate to ask for help in your PR comments

Remember, contributing to open source is a learning process, and the maintainers are here to help you succeed!

## Quick Reference

### First-time Contributors

1. Fork the repository
2. Set up your development environment
3. Find a "good first issue"
4. Make your changes
5. Run tests and formatting
6. Create a pull request

### Common Commands

```shell
# Setup (uv)
uv sync --extra dev
uv run pre-commit install

# Testing
uv run pytest tests/predict

# Formatting
uv run pre-commit run --all-files

# Add and commit
git add .
git commit -m "descriptive commit message"
```

