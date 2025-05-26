import os


def test_nav_files_exist():
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
        if ".md" in line:
            # Extract the markdown filename and clean it up
            md_file = line.strip().split(":")[-1].strip()
            # Remove list markers and quotes
            md_file = md_file.lstrip("- ").strip("'").strip('"')
            if md_file.endswith(".md"):
                md_files.append(md_file)

    # Check if files exist
    missing = []
    for file in md_files:
        if not os.path.exists(os.path.join(docs_dir, file)):
            missing.append(file)

    print("\nChecking files in:", docs_dir)
    print("Found MD files:", md_files)
    print("Missing files:", missing)

    assert not missing, f"Missing files: {missing}"
