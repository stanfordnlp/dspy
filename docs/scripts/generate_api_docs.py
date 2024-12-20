from pathlib import Path
import importlib
import inspect
import pkgutil
import shutil
import sys
from typing import Any


def get_module_contents(module):
    """Get all public classes and functions from a module."""
    contents_in_all = getattr(module, "__all__", None)

    contents = {}
    for name, obj in inspect.getmembers(module):
        if contents_in_all and name not in contents_in_all:
            continue
        if inspect.ismodule(obj) and obj.__name__.startswith(module.__name__) and not name.startswith("_"):
            contents[name] = obj
        elif (
            (inspect.isclass(obj) or inspect.isfunction(obj))
            and obj.__module__.startswith(module.__name__)
            and not name.startswith("_")
        ):
            contents[name] = obj
    return contents


def get_object_source_path(obj):
    """Get the source file path for an object."""
    try:
        return inspect.getfile(obj)
    except TypeError:
        return None


def get_original_name(obj):
    """Get the original name of an object if it's an alias."""
    if hasattr(obj, "__name__"):
        return obj.__name__
    return None


def generate_alias_section(name, obj, is_root=False):
    """Generate the alias section for a documentation file."""
    source_path = get_object_source_path(obj)
    if not source_path:
        return ""

    try:
        path_parts = Path(source_path).parts
        if "dspy" in path_parts:
            dspy_index = path_parts.index("dspy")
            rel_parts = path_parts[dspy_index + 1 :]

            module_parts = []
            for part in rel_parts:
                if part.endswith(".py"):
                    break
                if not part.startswith("_"):
                    module_parts.append(part)

            module_path = ".".join(module_parts)

            if module_path == "__init__" or not module_path:
                return ""

            if is_root:
                original_name = get_original_name(obj)
                if original_name and original_name != name:
                    return f"""## Alias

This object can also be accessed as `dspy.{original_name}`
"""
                return f"""## Alias

This object can also be accessed as `{module_path}.{name}`
"""
            else:
                return f"""## Alias

This object can also be accessed as `{module_path}.{name}`
"""
    except (ValueError, IndexError):
        return ""


def get_public_methods(cls):
    """Returns a list of all public methods in a class."""
    return [
        name
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name == "__call__" or not name.startswith("_")  # Exclude private and dunder methods
    ]


def generate_doc_page(name: str, module_path: str, obj: Any, is_root: bool = False) -> str:
    """Generate documentation page content for an object."""
    members_config = ""
    if inspect.isclass(obj):
        methods = get_public_methods(obj)
        if methods:
            methods_list = "\n".join(f"            - {method}" for method in methods)
            members_config = f"""
        members:
{methods_list}"""

    return f"""# {module_path}.{name}

::: {module_path}.{name}
    handler: python
    options:{members_config}
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false

{generate_alias_section(name, obj, is_root=is_root)}
"""


def generate_md_docs(output_dir: Path, excluded_modules=None):
    module = importlib.import_module("dspy")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_contents = get_module_contents(module)
    root_objects = {}

    # Generate docs for root-level objects, e.g. dspy.Predict, dspy.Example, etc.
    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            continue

        page_content = generate_doc_page(name, "dspy", obj, is_root=True)
        with open(output_dir / f"{name}.md", "w") as f:
            f.write(page_content)

        root_objects[f"{obj.__module__}.{name}"] = obj

    for submodule in pkgutil.iter_modules(module.__path__, prefix=f"{module.__name__}."):
        submodule_name = submodule.name.split(".")[-1]

        # Skip if this is a private module or not in __init__.py
        if submodule_name.startswith("_") or submodule_name not in init_contents:
            continue

        generate_md_docs_submodule(submodule.name, output_dir / submodule_name, root_objects, excluded_modules)


def generate_md_docs_submodule(module_path: str, output_dir: Path, root_objects=None, excluded_modules=None):
    if excluded_modules and module_path in excluded_modules:
        return

    try:
        module = importlib.import_module(module_path)
    except ImportError:
        print(f"Skipping {module_path} due to import error")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    init_contents = get_module_contents(module)

    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            continue

        full_name = f"{obj.__module__}.{name}"
        if full_name not in root_objects:
            # Only generate docs for objects that are not root-level objects.
            page_content = generate_doc_page(name, module_path, obj, is_root=False)
            with open(output_dir / f"{name}.md", "w") as f:
                f.write(page_content)

            root_objects[full_name] = obj

    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            generate_md_docs_submodule(f"{module_path}.{name}", output_dir / name, root_objects)


def remove_empty_dirs(path: Path):
    """Recursively remove empty directories."""
    for child in path.glob("*"):
        if child.is_dir():
            remove_empty_dirs(child)

    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


if __name__ == "__main__":
    api_dir = Path("docs/api")
    if api_dir.exists():
        shutil.rmtree(api_dir)

    excluded_modules = ["dspy.dsp"]
    generate_md_docs(api_dir, excluded_modules=excluded_modules)
    # Add final cleanup of empty directories
    remove_empty_dirs(api_dir)
