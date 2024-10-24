# DSPy Documentation

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Building docs locally

To build and test the documentation locally:

1. Navigate to the `docs` directory:
   ```bash
   cd docs
   ```

2. Install the necessary dependencies:
   ```bash
   npm install
   ```

3. Run the build command:
   ```bash
   npm run build
   ```

This will generate a static build of the documentation site in the `build` directory. You can then serve this directory to view the site locally. If you see the build failing make sure to fix it before pushing.

## Continuous Integration (CI) Build Checks

We have automated build checks set up in our CI pipeline to ensure the documentation builds successfully before merging changes. These checks:

1. Run the `npm run build` command
2. Verify that the build completes without errors
3. Help catch potential issues early in the development process

If the CI build check fails, please review your changes and ensure the documentation builds correctly locally before pushing updates.

## Contributing to the `docs` Folder

This guide is for contributors looking to make changes to the documentation in the `dspy/docs` folder. 

1. **Pull the up-to-date version of the website**: Please pull the latest version of the live documentation site via its subtree repository with the following command:

```bash
#Ensure you are in the top-level dspy/ folder
git subtree pull --prefix=docs https://github.com/krypticmouse/dspy-docs master
```

2. **Push your new changes on a new branch**: Feel free to add or edit existing documentation and open a PR for your changes. Once your PR is reviewed and approved, the changes will be ready to merge into main. 

3. **Updating the website**: Once your changes are merged to main, the changes would be reflected on live websites usually in 5-15 mins.