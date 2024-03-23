#!/bin/sh

git config --global --add safe.directory /workspaces/dspy

pip install poetry

poetry install --all-extras

sudo apt update

personalization_script="./.devcontainer/.personalization.sh"

# Developers can place a personalization script in the location specified above
# to further customize their dev container
if [ -f "$personalization_script" ]; then
    echo "File $personalization_script exists. Running the script..."
    chmod +x "$personalization_script"
    $personalization_script
fi
