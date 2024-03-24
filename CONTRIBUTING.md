# Contributing

## Setting-up

### Preferred method - VSCode Dev Container

VSCode dev containers are a great way to containerize not only the necessary requirements but also recommended IDE extensions as well as settings such as pre-commit hooks and linting preferences. Using this will allow you to jump in to the perfect DSPY contribution environment without having to do much. Additionally, you'll be able to contribute through the web browser using Github Codespaces!

To use our dev container:

1. Download Docker Desktop
2. Download VSCode
3. Within VSCode, install the Remote Development extension (ms-vscode-remote.vscode-remote-extensionpack)
4. Open the VSCode command palette (cmd / ctrl + shift + p)
5. Select `Dev Containers: Rebuild and Reopen in container`. A new VSCode window should open up and it should begin setting up your environment. Once it's done, you can open up a new terminal and start running tests or contributing!
6. To test that your environment is set up correctly, open a new terminal and run the command `pytest`. You should be able to run all tests and see them pass. Alternatively, you can open up the testing panel, which looks like a beaker, and click the play button to run all of our tests.
7. After the initial build, you should now be able to leave and re-enter the container any time without needing to rebuild. To do this, open the command palette and select `Dev Containers: Reopen in container`. This will not rebuild the container if you've done it correctly.

NOTE: If you use this method, your default shell will be the poetry shell which will contain all the necessary requirements in your terminal. You shouldn't need to prefix python commands with poetry as you're already using the correct poetry virtual environment.

### Alternative method

To run the tests, you need to first clone the repository.

Then install the package through poetry:
Note - You may need to install poetry. You likely will just need to run `pip install poetry`. See [here](https://python-poetry.org/docs/#installing-with-the-official-installer)

After installing poetry, use it to install our development requirements.

```bash
poetry install --with dev
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

This ends up looking like `feature(dspy): added new feature` or `enh(devcontainer): decreased size of image

```
^(break|build|ci|docs|feat|fix|perf|refactor|style|test|ops|hotfix|release|maint|init|enh|revert)\([a-z,A-Z,0-9,\-,\_,\/,:]+\)(:)\s{1}([\w\s]+)
```
