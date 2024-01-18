# Setting-up

## Crete new envirinment

```
conda create --name dspy python=3.11
```

or

```
python3 -m venv dspy
```

## Pre-commit hook

Before using pre-commit hook you need to install it in your python environment.

```
conda install -c conda-forge pre-commit
```

go to the root folder and then activate it as follows (it will first download all required dependencies):

```
pre-commit install
```

> Pre-commit hooks will attept to fix all your files and so you will need to (add + commit) them once the fixes are done !

#### Optional:

Generally the pre-commit will run automatically before each of your commit,
but you can also manually trigger it, as follows:

```
pre-commit run --all-files
```

## Commit Message format:

Commit message format must be respected, with the followint regex:

```
^(break|build|ci|docs|feat|fix|perf|refactor|style|test|ops|hotfix|release|maint|init|enh|revert)\([a-z,A-Z,0-9,\-,\_,\/,:]+\)(:)\s{1}([\w\s]+)
```
