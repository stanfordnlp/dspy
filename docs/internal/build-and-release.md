# Build & Release Workflow Implementation

The [build_and_release](https://github.com/stanfordnlp/dspy/blob/main/.github/workflows/build_and_release.yml) workflow automates deployments of dspy-ai to pypi. For a guide to triggering a release using the workflow, refer to [release checklist](release-checklist.md).

## Overview

At a high level, the workflow works as follows: 

1. Maintainer of the repo pushes a tag following [semver](https://semver.org/) versioning for the new release.
2. This triggers the github action which extracts the tag (the version)
3. Builds and publishes a release on [test-pypi](https://test.pypi.org/project/dspy-ai-test/)
4. Uses the test-pypi release to run build_utils/tests/intro.py with the new release as an integration test. Note intro.py is a copy of the intro notebook.
5. Assuming the test runs successfully, it pushes a release to [pypi](https://pypi.org/project/dspy-ai/). If not, the user can delete the tag, make the fixes and then push the tag again. Versioning for multiple releases to test-pypi with the same tag version is taken care of by the workflow by appending a pre-release identifier, so the user only needs to consider the version for pypi. 
6. (Currently manual) the user creates a release and includes release notes, as described in docs/docs/release-checklist.md

## Implementation Details

The workflow executes a series of jobs in sequence: 
- extract-tag
- build-and-publish-test-pypi
- test-intro-script
- build-and-publish-pypi

#### extract-tag
Extracts the tag pushed to the commit. This tag is expected to be the version of the new deployment. 

#### build-and-publish-test-pypi
Builds and publishes the package to test-pypi.
1. Determines the version that should be deployed to test-pypi. There may be an existing deployment with the version specified by the tag in the case that a deployment failed and the maintainer made some changes and pushed the same tag again (which is the intended usage). The following logic is implemented [test_version.py](https://github.com/stanfordnlp/dspy/blob/main/build_utils/test_version.py)
    1. Load the releases on test-pypi
    1. Check if there is a release matching our current tag
        1. If not, create a release with the current tag
        1. If it exists, oad the latest published version (this will either be the version with the tag itself, or the tag + a pre-release version). In either case, increment the pre-release version.
1. Updates the version placeholder in [setup.py](https://github.com/stanfordnlp/dspy/blob/main/setup.py) to the version obtained in step 1.
1. Updates the version placeholder in [pyproject.toml](https://github.com/stanfordnlp/dspy/blob/main/pyproject.toml) to the version obtained in step 1.
1. Updates the package name placeholder in [setup.py](https://github.com/stanfordnlp/dspy/blob/main/setup.py) to  `dspy-ai-test`*
1. Updates the package name placeholder in [pyproject.toml](https://github.com/stanfordnlp/dspy/blob/main/pyproject.toml) to `dspy-ai-test`*
1. Builds the binary wheel
1. Publishes the package to test-pypi. 


#### test-intro-script
Runs the pytest containing the intro script as an integration test using the package published to test-pypi. This is a validation step before publishing to pypi.
1. Uses a loop to install the version just published to test-pypi as sometimes there is a race condition between the package becoming available for installation and this job executing.
2. Runs the test to ensure the package is working as expected. 
3. If this fails, the workflow fails and the maintainer needs to make a fix and delete and then recreate the tag.

#### build-and-publish-pypi
Builds and publishes the package to pypi.

1. Updates the version placeholder in [setup.py](https://github.com/stanfordnlp/dspy/blob/main/setup.py) to the version obtained in step 1.
1. Updates the version placeholder in [pyproject.toml](https://github.com/stanfordnlp/dspy/blob/main/pyproject.toml) to the version obtained in step 1.
1. Updates the package name placeholder in [setup.py](https://github.com/stanfordnlp/dspy/blob/main/setup.py) to  `dspy-ai`*
1. Updates the package name placeholder in [pyproject.toml](https://github.com/stanfordnlp/dspy/blob/main/pyproject.toml) to `dspy-ai`*
1. Builds the binary wheel
1. Publishes the package to pypi.


\* The package name is updated by the worfklow to allow the same files to be used to build both the pypi and test-pypi packages.