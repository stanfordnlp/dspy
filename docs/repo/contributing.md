# ⚙️ Setting-up working envirionment

## 💻 Env Setup

```
conda create --name dspy python=3.11
```

or

```
python3 -m venv dspy
```

## 🚀 Pre-commit hook

Before using pre-commit hook you need to install it in your python environment.

```
conda install -c conda-forge pre-commit
```

go to the root folder and then activate it as follows (it will first download all required dependencies):

```
pre-commit install
```

> Pre-commit hooks will attept to fix all your files and so you will need to (add + commit) them once the fixes are done !

!!! info "Optionally"

    Generally the pre-commit will run automatically before each of your commit,
    but you can also manually trigger it, as follows:

    ```python
    pre-commit run --all-files
    ```

## 📝 Commit with Style

Use standarized commit message:

`{LABEL}(ACRONYM): {message}`

This is very important for the automatic releases and a clean history on the `main` branch.

!!! Labels-types

    | Label | Usage |
    | ----- | ----- |
    | break| `break` is used to identify changes related to old compatibility or functionality that breaks the current usage (major) |
    | feat | `feat` is used to identify changes related to new backward-compatible abilities or functionality (minor) |
    | init | `init` is used to indentify the starting related to the project (minor) |
    | enh | `enh` is used to indentify changes related to amelioration of abilities or functionality (patch) |
    | build | `build` (also known as `chore`) is used to identify **development** changes related to the build system (involving scripts, configurations, or tools) and package dependencies (patch) |
    | ci | `ci` is used to identify **development** changes related to the continuous integration and deployment system - involving scripts, configurations, or tools (minor) |
    | docs | `docs`  is used to identify documentation changes related to the project; whether intended externally for the end-users or internally for the developers (patch) |
    | perf | `perf`  is used to identify changes related to backward-compatible **performance improvements** (patch) |
    | refactor | `refactor` is used to identify changes related to modifying the codebase, which neither adds a feature nor fixes a bug - such as removing redundant code, simplifying the code, renaming variables, etc.<br />i.e. handy for your wip ; ) (patch) |
    | style | `style`  is used to identify **development** changes related to styling the codebase, regardless of the meaning - such as indentations, semi-colons, quotes, trailing commas, and so on (patch) |
    | test | `test` is used to identify **development** changes related to tests - such as refactoring existing tests or adding new tests. (minor) |
    | fix | `fix`  is used to identify changes related to backward-compatible bug fixes. (patch) |
    | ops | `ops` is used to identify changes related to deployment files like `values.yml`, `gateway.yml,` or `Jenkinsfile` in the **ops** directory. (minor) |
    | hotfix | `hotfix` is used to identify **production** changes related to backward-compatible bug fixes (patch) |
    | revert | `revert` is used to identify backward changes (patch) |
    | maint | `maint` is used to identify **maintenance** changes related to project (patch) |

```

```
