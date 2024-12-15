from pathlib import Path
import yaml
import mkdocs_gen_files
import shutil
import os


# Custom YAML constructor to handle !python/name tags
class CustomLoader(yaml.SafeLoader):
    pass


class CustomDumper(yaml.SafeDumper):
    pass


def construct_undefined(loader, suffix, node):
    """Handle all undefined tags"""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)


# Add constructors for handling any undefined tags
CustomLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", construct_undefined)
CustomLoader.add_multi_constructor("!", construct_undefined)


def build_nav_structure(directory: Path, base_path: Path) -> dict:
    """Recursively build navigation structure for a directory."""
    nav = {}

    # First handle all MD files in current directory
    for path in sorted(directory.glob("*.md")):
        name = path.stem
        nav[name] = str(path.relative_to(base_path))

    # Then handle all subdirectories
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            sub_nav = build_nav_structure(path, base_path)
            if sub_nav:  # Only add non-empty directories
                nav[path.name] = sub_nav

    return nav


# First read the existing mkdocs.yml
with open("mkdocs.yml", "r") as f:
    lines = f.readlines()

    # Find the nav section start and theme section start
    nav_start = -1
    theme_start = -1

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

# Generate the API nav structure
api_nav = {}
api_path = Path("docs/api")

# Process each top-level module directory
for dir_path in sorted(api_path.iterdir()):
    if dir_path.is_dir():
        category = dir_path.name
        api_nav[category] = build_nav_structure(dir_path, Path("docs"))


def format_nav_section(nav_dict, indent_level=2):
    """Convert dictionary to properly indented nav section"""
    lines = []
    indent = "    " * indent_level

    for key, value in sorted(nav_dict.items()):
        if isinstance(value, dict):
            # This is a section
            lines.append(f"{indent}- {key}:")
            lines.extend(format_nav_section(value, indent_level + 1))
        else:
            # This is a file
            lines.append(f"{indent}- {key}: {value}")

    return lines


# Create API section
api_section = ["    - API Reference:"]
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
