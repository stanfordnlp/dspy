from pathlib import Path

INDEX_NAME = {
    "models": "Models",
    "primitives": "Primitives",
    "signatures": "Signatures",
    "adapters": "Adapters",
    "modules": "Modules",
    "evaluation": "Evaluation",
    "optimizers": "Optimizers",
    "utils": "Utils",
    "tools": "Tools",
    "experimental": "Experimental",
}


def build_nav_structure(directory: Path, base_path: Path) -> dict:
    """Recursively build navigation structure for a directory."""
    nav = {}

    # Get all items in current directory
    items = sorted(directory.iterdir())

    for path in items:
        if path.suffix == ".md":
            name = path.stem
            nav[name] = str(path.relative_to(base_path))
        elif path.is_dir() and path.name == "GEPA":
            nav["GEPA"] = {
                "2. GEPA Advanced": "api/optimizers/GEPA/GEPA_Advanced.md",
                "1. GEPA Overview": "api/optimizers/GEPA/overview.md",
            }

    return nav


def format_nav_section(nav_dict, indent_level=2):
    """Convert dictionary to properly indented nav section"""
    lines = []
    indent = "    " * indent_level

    module_navs = []
    file_navs = []
    for key, value in sorted(nav_dict.items()):
        if isinstance(value, dict):
            # This is a section
            module_navs.append(f"{indent}- {key}:")
            module_navs.extend(format_nav_section(value, indent_level + 1))
        else:
            # This is a file
            file_navs.append(f"{indent}- {key}: {value}")

    # Put submodules' nav items before file nav items. e.g., `dspy.evaluate` before `dspy.ChainOfThought`
    # in the nav section.
    lines.extend(module_navs)
    lines.extend(file_navs)

    return lines


def read_mkdocs_sections(filename: str = "mkdocs.yml"):
    """Read and parse the mkdocs.yml file into sections."""
    with open(filename, "r") as f:
        lines = f.readlines()

    nav_start = -1
    theme_start = -1

    # Find section boundaries
    for i, line in enumerate(lines):
        if line.strip() == "nav:":
            nav_start = i
        elif line.strip() == "theme:":
            theme_start = i
            break

    # Split content into sections
    pre_nav = lines[: nav_start + 1]  # Include the 'nav:' line
    nav_content = []
    post_theme = lines[theme_start:]  # Start from 'theme:' line

    # Extract nav content excluding API Reference
    i = nav_start + 1
    while i < theme_start:
        line = lines[i]
        if line.strip() == "- API Reference:":
            # Skip this line and all indented lines that follow
            i += 1
            while i < theme_start and (not lines[i].strip() or lines[i].startswith(" " * 8)):
                i += 1
        else:
            nav_content.append(line)
            i += 1

    return pre_nav, nav_content, post_theme


def generate_api_nav():
    """Generate the API navigation structure."""
    api_nav = {}
    api_path = Path("docs/api")

    for dir_path in sorted(api_path.iterdir()):
        if dir_path.is_dir():
            category = INDEX_NAME[dir_path.name]
            api_nav[category] = build_nav_structure(dir_path, Path("docs"))

    return api_nav


def main():
    """Main function to generate the API documentation summary."""
    # Read existing mkdocs.yml sections
    pre_nav, nav_content, post_theme = read_mkdocs_sections()

    # Generate API navigation structure
    api_nav = generate_api_nav()

    # Create API section
    api_section = ["    - API Reference:"]
    api_section.append("        - API Reference: api/index.md")
    api_section.extend(format_nav_section(api_nav))
    api_section.append("")  # Add empty line before theme section

    # Write back to mkdocs.yml
    with open("mkdocs.yml", "w") as f:
        # Write pre-nav content
        f.writelines(pre_nav)
        # Write nav content
        f.writelines(nav_content)
        # Add API section
        f.write("\n".join(api_section) + "\n")
        # Write post-theme content
        f.writelines(post_theme)


if __name__ == "__main__":
    main()
