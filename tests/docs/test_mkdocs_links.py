import os


def test_nav_sources_exist():
    # Read mkdocs.yml
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "docs")
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "mkdocs.yml")

    # Read file and extract nav section
    with open(yaml_path) as f:
        content = f.read()

    # Find nav section
    nav_start = content.find("nav:")
    lines = content[nav_start:].split("\n")

    # Get markdown files
    md_files = []
    for line in lines:
        if ".md" in line and "*.md" not in line:  # Ignore patterns in llmstxt: plugin
            # Extract the markdown filename and clean it up
            md_file = line.strip().split(":")[-1].strip()
            # Remove list markers and quotes
            md_file = md_file.lstrip("- ").strip("'").strip('"')
            if md_file.endswith(".md"):
                md_files.append(md_file)

    # Notebook and Python pages are converted to Markdown in the disposable
    # Zensical project, so their nav entries name the generated .md path.
    missing = []
    for file in md_files:
        source = os.path.join(docs_dir, file)
        generated_sources = [os.path.splitext(source)[0] + suffix for suffix in (".ipynb", ".py")]
        if not os.path.exists(source) and not any(os.path.exists(candidate) for candidate in generated_sources):
            missing.append(file)

    print("\nChecking files in:", docs_dir)
    print("Found MD files:", md_files)
    print("Missing files:", missing)

    assert not missing, f"Missing files: {missing}"
