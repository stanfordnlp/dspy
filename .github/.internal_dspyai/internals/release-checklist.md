# Release Checklist

* [ ] For DSPy 3.4, announce the [3.5 LM-interface cutoff](https://dspy.ai/community/normalized-lm-api-migration/#the-35-cutoff) in the release notes: legacy `forward()`/`aforward()` integrations, both legacy-engine wrappers, custom `complete_legacy()` shortcuts, and OpenAI-style `messages=` calls are deprecated. Include the request/response migration examples and clarify that `lm("hello")` stays supported.
* [ ] Before releasing DSPy 3.5, verify that the deprecated LM interfaces and temporary adapter-message warning marker are removed, all adapters and engines use lm15 requests/responses, and examples/tests no longer rely on the old interfaces.

* [ ] Before tagging, refresh the bundled model capability/pricing metadata from a selected **released** LiteLLM version. From the repository root, run `python scripts/update_model_metadata.py VERSION` (replace `VERSION` with the release you reviewed; the script's default is pinned, not "latest"). This downloads metadata and license files without importing LiteLLM or making model calls.
    * Review and commit `dspy/clients/model_metadata/snapshot.json.gz`, `provenance.json`, and `LICENSE` before creating the release tag. Check the recorded version and source hashes; do not regenerate these files during the release build itself.
    * Run `LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest tests/clients/test_native_metadata.py` against the updated snapshot. Review changed capabilities/prices rather than adjusting expectations just to make the tests pass.
* [ ] On `main` Create a git tag with pattern X.Y.Z where X, Y, and Z follow the [semver pattern](https://semver.org/). Then push the tag to the origin git repo (github).
    * ```bash
      git tag X.Y.Z
      git push origin --tags
      ```
    * This will trigger the github action to build and release the package.
* [ ] Confirm the tests pass and the package has been published to pypi.
* [ ] Merge the auto-generated version bump PR that is opened against `main`.
    * If the tests fail, you can remove the tag from your local and github repo using:
    ```bash
    git push origin --delete X.Y.Z # Delete on GitHub
    git tag -d X.Y.Z # Delete locally
    ```
    * Fix the errors and then repeat the steps above to recreate the tag locally and push to GitHub to restart the process.
    * Note that the github action takes care of incrementing the release version on test-pypi automatically by adding a pre-release identifier in the scenario where the tests fail and you need to delete and push the same tag again. 
* [ ] [Create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) 
* [ ] Add release notes. You can make use of [automatically generated release notes](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes)
* If creating a new release for major or minor version:
    * [ ] Create a new release branch with the last commit and name it 'release/X.Y`
    * [ ] [Update the default branch](https://docs.github.com/en/organizations/managing-organization-settings/managing-the-default-branch-name-for-repositories-in-your-organization) on the github rep to the new release branch.

### Prerequisites

The automation requires a [trusted publisher](https://docs.pypi.org/trusted-publishers/) to be set up on both the pypi and test-pypi packages. If the package is migrated to a new project, please follow the [steps](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) to create a trusted publisher. If you have no releases on the new project, you may have to create a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) to allow the first automated deployment. 