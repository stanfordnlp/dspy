#!/bin/bash

# Exit if insufficient arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: ./sync_to_public.sh <branch-name> [--squash]"
    exit 1
fi

BRANCH_NAME=$1
SQUASH_COMMITS=false

# Check if the squash option is provided
if [ "$#" -eq 2 ] && [ "$2" == "--squash" ]; then
    SQUASH_COMMITS=true
fi

# Variables
PRIVATE_REMOTE="origin"
PUBLIC_REMOTE="upstream"
CURRENT_DATE=$(date +%Y%m%d%H%M%S)
TEMP_BRANCH="temp-public-sync-$CURRENT_DATE"

# Ensure we are on the specified branch and have the latest changes
git checkout $BRANCH_NAME
git pull $PRIVATE_REMOTE $BRANCH_NAME

# If squash option is true, squash the commits
if [ "$SQUASH_COMMITS" = true ]; then
    # Count the number of commits since the last public sync
    NUM_COMMITS=$(git rev-list --count $PUBLIC_REMOTE/$BRANCH_NAME..HEAD)

    # Squash the last NUM_COMMITS into one
    if [ "$NUM_COMMITS" -gt 1 ]; then
        git reset --soft HEAD~$NUM_COMMITS && git commit
    fi
fi

# Create a new temporary branch
git checkout -b $TEMP_BRANCH

# Remove private_* directories and commit
git rm -r private_* 2> /dev/null
git commit -m "Remove private content for public sync"

# Push the temporary branch to the corresponding branch of the public repo
git push $PUBLIC_REMOTE $TEMP_BRANCH:$BRANCH_NAME

# Return to the specified branch and delete the temporary branch
git checkout $BRANCH_NAME
git branch -D $TEMP_BRANCH

echo "Sync completed successfully!"
