# Contibuting

## Finding Issues

### Bounty Board

The bounty board will have various features, issues, and requests that are up for grabs. We are still working on this. Come to the discord and ask for the current bounties.

See the spreadsheet [here](https://docs.google.com/spreadsheets/d/1psHSfFXENAxhQTd5veKRzKydVubD2Ov62aKQHiYC-CQ/edit?usp=sharing) for the current bounties.

## Setting-up

To run the tests, you need to first clone the repository.

Then install the package through poetry:
Note - You may need to install poetry. See [here](https://python-poetry.org/docs/#installing-with-the-official-installer)

```bash
poetry install --with test
```

## Testing

To run the all tests, or a specific test suite, use the following commands:

```bash
poetry run pytest
poetry run pytest tests/PATH_TO_TEST_SUITE
```

If you are changing CI actions, you can use the [act](https://nektosact.com/introduction.html) tool to test the CI locally.

Example for testing the push action:
You may need the `--container-architecture linux/amd64` flag if you are on an M1/2 mac.

```bash
 act push
```

## Commit Message format

Commit message format must be respected, with the following regex:

This ends up looking like feature(dspy): added new feature

```
^(break|build|ci|docs|feat|fix|perf|refactor|style|test|ops|hotfix|release|maint|init|enh|revert)\([a-z,A-Z,0-9,\-,\_,\/,:]+\)(:)\s{1}([\w\s]+)
```
