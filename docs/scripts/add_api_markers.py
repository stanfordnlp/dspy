from pathlib import Path


def add_markers_to_file(file_path: Path):
    """Add API reference markers to an existing documentation file."""
    if not file_path.exists():
        return

    content = file_path.read_text()

    # Skip if the file already has markers
    if "<!-- START_API_REF -->" in content:
        return

    # Find the API reference section marked by :::
    api_start = content.find(":::")
    if api_start == -1:
        return  # No API section found

    # Find where the API section ends (next non-indented line after :::)
    lines = content[api_start:].split("\n")
    api_end = 0
    for i, line in enumerate(lines):
        if i > 0 and (not line.strip() or not line.startswith(" ")):
            api_end = api_start + sum(len(l) + 1 for l in lines[:i])
            break
    else:
        api_end = len(content)

    # Split content into sections
    pre_content = content[:api_start].rstrip()
    api_content = content[api_start:api_end].rstrip()
    post_content = content[api_end:].lstrip()

    # Combine with new markers
    new_content = f"{pre_content}\n\n<!-- START_API_REF -->\n{api_content}\n<!-- END_API_REF -->"
    if post_content:
        new_content += f"\n\n{post_content}"
    new_content += "\n"

    # Write back to file
    file_path.write_text(new_content)


def main():
    """Add API reference markers to all documentation files."""
    api_dir = Path("docs/api")
    if not api_dir.exists():
        print("API documentation directory not found")
        return

    # Process all markdown files in the API directory and its subdirectories
    for file_path in api_dir.rglob("*.md"):
        if file_path.name == "index.md":
            continue  # Skip index files
        print(f"Processing {file_path}")
        add_markers_to_file(file_path)


if __name__ == "__main__":
    main()
