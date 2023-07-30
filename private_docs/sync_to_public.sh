#!/bin/bash

# Ensure the script exits on any error
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: ./sync_to_public.sh <branch_name> [--squash]"
    exit 1
fi

BRANCH=$1
SQUASH=$2

# Ensure we are in the correct branch
git checkout $BRANCH

# Create a temporary branch for this sync process
TEMP_BRANCH="temp-public-sync-$(date +%Y%m%d%H%M%S)"
git checkout -b $TEMP_BRANCH

# Pull any new changes from the public repo (this should ideally be a fast-forward merge)
git pull upstream $BRANCH

# List the private directories to be removed
echo "The following private directories will be removed:"
find . -type d -name 'private_*'

# Ask for confirmation
read -p "Are you sure you want to remove these directories for public sync? [y/N] " response

if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]
then
    # Remove the private directories 
    find . -type d -name 'private_*' -exec rm -r {} +

    # Stage the removal changes
    git add -A

    # Commit the removal of private directories (this will be a no-op if no private directories were present)
    git commit -m "Prepare for public sync" || echo "No private directories to remove"
else
    echo "Aborted removing private directories."
    exit 1
fi

# If squash option is provided, squash all the commits 
if [ "$SQUASH" == "--squash" ]; then
    echo "Squashing commits..."
    git reset $(git commit-tree HEAD^{tree} -m "Squash all commits for public sync")
fi

# Push the changes to the public repo
git push upstream $TEMP_BRANCH:$BRANCH

# Return to the original branch and delete the temporary branch
git checkout $BRANCH
git branch -D $TEMP_BRANCH

echo "Sync completed successfully!"
