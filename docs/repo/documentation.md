# 📃 Documentation

We are using MkDocs to build the documentation, you can find more info about all possibilites here: [Examples](https://squidfunk.github.io/mkdocs-material/reference/#setting-the-page-title). It is basically a combination of Markdown and Mermaid for graph.

!!! info
You can read more about [Mermaid](https://github.com/mermaid-js/mermaid) with all its possibilities !
If you would like to test your mermaid FlowChart online without having to install any required libraries, you can refere to [Online Schema Editor](https://mermaid-js.github.io/mermaid-live-editor)

## ➕ Extending the documentation

- [x] start from the `dev` branch and create a new branch with the correct naming convention, see: [how to contribute](contributing.md)

- [x] add additional '.md' files to the documentation directory: `/docs`

- [x] add the new entry into the navigation bar and connect it with your `md` file. This can be done in: [`root/mkdocs.yml`](mkdocs.yml)

- [x] You can interactively test the documentation locally by using the following command: `mkdocs serve`

  > You will need to have all local docs-related requirements installed (see: [tool.poetry.group.doc.dependencies]):

  ```python
  mkdocs = ">=1.5.3"
  mkdocs-material = ">=9.0.6"
  mkdocs-material-extensions = ">=1.3.1"
  mkdocs-gen-files = "^0.5.0"
  mkdocstrings-python = "^1.7.5"
  mkdocstrings = {extras = ["python"], version = ">=0.20.0"}
  mike = ">=2.0.0"
  ```

- [x] Once you are done, create a new Merge Request to the `dev` branch.

- [x] When your MR gets approved merge it into the `dev` following the well know conventions [how to contribute](contributing.md)

- [x] New documentation will be automatically deployed once your MR gets merged !

!!! warning
In some cases you may need to deploy the new doc to Github-pages immediatly, this can be done using the following command: `mkdocs gh-deploy` (while being in a right venv)

## 🔍 Documenting code

Documenting code is done using dedicated docstrings, which are then automatically parsed and converted into the documentation.

In order to document your code, you need to use the following syntax:

```python
# 🗺️ PARAGRAPG NAME

::: dspy.predict.predict.Predict
    handler: python
    options:
        show_root_heading: true
        show_source: true
```

and the Predict class documentation needs to foollow the Google Style Guide, see: [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

!!! example

    ```python
    """Send a static HTML report along with a message to the Slack channel.

        Args:
            project_name (str): The name of the project for which the job was submitted.
            html_file_path (str): The file path of the HTML report to be sent.
            message (str): The message to send along with the HTML report.

        Example:
            ```python
            from theparrot.notify import SlackBot

            bot = SlackBot()
            bot.send_html_report(
                html_file_path="path/to/your/report.html",
                message="Check out this report!",
            )
            ```
        """
    ```

This approach allows to handle documentation directly from the code, which is a great way to keep it up to date.
It also allows to version the documentation, which is a great way to keep track of changes and handle multiple versions of the package.
