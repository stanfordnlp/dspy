#!/bin/sh

set -e  # Exit immediately if a command exits with a non-zero status.

git config --global --add safe.directory /workspaces/dspy

pip install poetry==1.7.1
poetry config installer.max-workers 4

poetry install --with dev

sudo apt update
sudo apt-get -y install python3-distutils

poetry run pre-commit install --install-hooks

personalization_script="./.devcontainer/.personalization.sh"

# Developers can place a personalization script in the location specified above
# to further customize their dev container
if [ -f "$personalization_script" ]; then
    echo "File $personalization_script exists. Running the script..."
    chmod +x "$personalization_script"
    $personalization_script
fi
