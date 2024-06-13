from rich.console import Console
from rich.text import Text

text = Text()
console = Console()

text.append(
    "Be sure to check that you have the Poetry Python interpreter selected in VSCode otherwise"
    " your terminal will not include the necessary packages to develop. If you have the correct interpreter"
    " configured, you'll see an interpreter that looks like this: ",
    style="bold yellow",
)
text.append("3.9.18 ('.venv': Poetry)", style="code")
text.append(
    ". If you have previously selected another Python interpreter for the VSCode workspace, you may need to select"
    " the Poetry Python interpreter.\n\nTo do so, follow the instructions at the following link and choose the Poetry"
    " virtual environment interpreter.",
    style="bold yellow",
)

console.print(text)
console.print(
    "VSCode Python Interpreter Documentation\n",
    style="link https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters",
)
