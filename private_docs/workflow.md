## Dual Repository Workflow: Public and Private Repositories on GitHub

### 1) Introduction

When developing sensitive projects with the intent to periodically release public versions, a dual repository workflow offers the advantage of private development while ensuring periodic transparent releases. This document details the steps and practices for managing both public and private repositories on GitHub, with a specific focus on ensuring that certain private contents never get pushed to the public repository.

### 2) Setup [do this once]

You can do this once (i.e., every time you need a new clone to work from).

Step 1: **Initialization**

- Private Repository: A repository (`okhat/dsp-private`) where all development work occurs, including potentially sensitive data.
- Public Repository: A repository (`stanfordnlp/dsp`) where sanitized, non-sensitive work is pushed for public consumption.

Step 2: **Clone the Private Repository**

```bash
git clone https://github.com/okhat/dsp-private.git
cd dsp-private
```


Step 3: **Add the Public Repository as an Upstream Remote**

```bash
git remote add upstream https://github.com/stanfordnlp/dsp.git
```

### 3) Development Workflow [relevant for what you do every day, especially Steps 2 and 3]

This is what you do in the private repo day to day.

Step 1: **Pull Latest Changes from Private Repository**

Ensure you're always working with the latest data from the private repo:

```bash
git checkout <branch-name>
git pull origin <branch-name>
```

Replace `<branch-name>` with the name of the branch you are working on.

Step 2: **Development** (this is the actual work!)

Conduct your development as usual.

When adding sensitive or private content, make sure to place it within directories prefixed by `private_*/`.

We already have `private_docs/` as an example.

Step 3: **Commit Changes in Private Repository**

```bash
git add .
git commit -m "Your descriptive commit message"
git push origin <branch-name>
```

Replace `<branch-name>` with the name of the branch you are working on.

### 4) Pulling Updates from the Public Repository [you do this when needed]

If the public repository receives contributions, you'll want to incorporate them into your private repository:

```bash
git checkout <branch-name>
git pull upstream <branch-name>  # pull and merge anything from public upstream
git push origin <branch-name>   # push now any recent things to the private origin
```

This is important before syncing back to the public repo. It ensures that both repositories are in sync and that you're working with the most recent changes.

You may sometimes do this with branches other than `main`.


### 5) Syncing to Public Repository [you do this when needed]

Step 1: **Use the Sync Script**

This is located in the `private_docs/` directory (i.e., here).

```bash
# First, ensure you have the latest changes from the public repo 
# and deal with any potential merge conflicts.
git checkout <branch-name>
git pull upstream <branch-name>

# Once everything is up-to-date and conflicts (if any) are resolved, 
# proceed to sync your changes to the public repo.
./sync_to_public.sh <name_of_branch> [--squash]
```

The specified `<branch-name>` could be `main`, but perhaps you're working on a feature or fix and want to merge from/to a specific branch like `feature_branch_name`. In such cases, replace `<branch-name>` with the name of your desired branch.

You can also optionally use the `--squash` argument. When used, this will squash all recent commits from the private repo into a single commit before syncing to the public repo. This is particularly useful when you want a cleaner and more consolidated commit history in the public repository.

We can agree on the better approach on a case-by-case basis.

This script will:
- Pull the latest changes from the private repository.
- Create a temporary branch with a unique name.
- Remove the `private_*` directories from this temporary branch.
- Push this sanitized temporary branch to the (selected) branch of the public repository.
- Return you to the (selected) branch of your private repository and delete the temporary branch.


#### FAQ: Is this squash safe, *IF* we follow the pull upstream sequence?

The `--squash` option in the provided script squashes commits in the private repo, but it does this in a temporary branch. So, the original commit history in the branch you're working on (e.g., main, feature_branch_name, etc.) in the private repo remains intact.

Here's a breakdown of how the squashing process in the script works:

- You check out the branch you want to sync (e.g., main).
- The script then creates a temporary branch off of your current branch.
- Within this temporary branch, the script squashes the commits.
- The changes (now squashed into one commit) are pushed to the public repo from this temporary branch.
- The script then deletes the temporary branch and returns you to your original branch.
- Your original branch in the private repo remains unchanged, so all your detailed commit history is still there.

In short: Yes, it's safe. If you decide later that you didn't want things squashed in the public repo, you can always sync again from the private repo without the --squash option, as the original commits remain intact in your private repo.


Step 2: **Review Public Repo**

After the script has been run, always take a moment to check the public repository on GitHub to ensure everything looks as expected.



### 6) Conclusion

This dual-repo system allows you to maintain a private workspace with potentially sensitive or incomplete work while providing the ability to periodically release sanitized versions to the public. Using the provided script ensures that sensitive directories (private_*) remain in the private repo and never get exposed to the public.

Remember to always backup important data, understand the implications of each step in the workflow, and maintain regular checks to ensure the integrity of both repositories.



