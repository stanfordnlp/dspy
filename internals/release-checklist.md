# Release Checklist

* [ ] On `main` Create a git tag with pattern X.Y.Z where X, Y, and Z follow the [semver pattern](https://semver.org/). Then push the tag to the origin git repo (github).
    * ```bash
      git tag X.Y.Z
      git push origin --tags
      ```
    * This will trigger the github action to build and release the package.
* [ ] Confirm the tests pass and the package has been published to pypi.
    * If the tests fail, you can remove the tag from your local and github repo using:
    ```bash
    git push origin --delete X.Y.Z # Delete on Github
    git tag -d X.Y.Z # Delete locally
    ```
    * Fix the errors and then repeat the steps above to recreate the tag locally and push to Github to restart the process.
    * Note that the github action takes care of incrementing the release version on test-pypi automatically by adding a pre-release identifier in the scenario where the tests fail and you need to delete and push the same tag again. 
* [ ] [Create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) 
* [ ] Add release notes. You can make use of [automatically generated release notes](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes)
* If creating a new release for major or minor version:
    * [ ] Create a new release branch with the last commit and name it 'release/X.Y`
    * [ ] [Update the default branch](https://docs.github.com/en/organizations/managing-organization-settings/managing-the-default-branch-name-for-repositories-in-your-organization) on the github rep to the new release branch.

### Prerequisites

The automation requires a [trusted publisher](https://docs.pypi.org/trusted-publishers/) to be set up on both the pypi and test-pypi packages. If the package is migrated to a new project, please follow the [steps](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) to create a trusted publisher. If you have no releases on the new project, you may have to create a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) to allow the first automated deployment. 