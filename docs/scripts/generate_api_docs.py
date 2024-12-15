from pathlib import Path
import importlib
import inspect
import shutil
import ast


def get_module_contents(module):
    """Get all public classes and functions from a module."""
    contents = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith("_"):
            module_name = inspect.getmodule(obj).__name__
            if module_name.startswith("dspy."):
                contents.append((name, obj))
    return contents


def get_imported_submodules(module_path):
    """Get submodules that are imported in __init__.py."""
    try:
        module_init = Path(importlib.util.find_spec(module_path).origin)
        if not module_init.name == "__init__.py":
            return []

        with open(module_init, "r") as f:
            tree = ast.parse(f.read())

        submodules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith(f"{module_path}."):
                        submodule = name.name
                        if submodule != module_path and not any(
                            submodule.startswith(existing + ".") for existing in submodules
                        ):
                            submodules.add(submodule)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(module_path):
                    submodule = node.module
                    if submodule != module_path and not any(
                        submodule.startswith(existing + ".") for existing in submodules
                    ):
                        submodules.add(submodule)
                elif node.level > 0:  # relative imports
                    base = module_path.split(".")
                    if node.level > len(base):
                        continue
                    parent = ".".join(base[: -node.level])
                    if node.module:
                        submodule = f"{parent}.{node.module}"
                        if submodule != module_path and not any(
                            submodule.startswith(existing + ".") for existing in submodules
                        ):
                            submodules.add(submodule)

        return [m for m in submodules if m.startswith(module_path)]
    except Exception as e:
        print(f"Warning: Error processing {module_path}: {e}")
        return []


def generate_object_doc(name: str, obj, output_dir: Path):
    """Generate documentation for a single object (class or function)."""
    original_module = inspect.getmodule(obj).__name__

    page_content = f"""# {name}

::: {original_module}.{name}
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_section_style: google
        show_root_full_path: false
        show_object_full_path: false
        separate_signature: false
"""
    with open(output_dir / f"{name}.md", "w") as f:
        f.write(page_content)


def generate_module_docs(module_path: str, output_dir: Path):
    """Generate API documentation for a module and its submodules."""
    # Import the module
    module = importlib.import_module(module_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all public classes and functions
    contents = get_module_contents(module)

    # Generate individual pages for each class and function
    for name, obj in contents:
        generate_object_doc(name, obj, output_dir)

    # Handle submodules that are imported in __init__.py
    for submodule_path in get_imported_submodules(module_path):
        submodule_name = submodule_path.split(".")[-1]
        submodule_dir = output_dir / submodule_name
        generate_module_docs(submodule_path, submodule_dir)


if __name__ == "__main__":
    api_dir = Path("docs/api")
    if api_dir.exists():
        shutil.rmtree(api_dir)

    generate_module_docs("dspy.datasets", Path("docs/api/datasets"))
    generate_module_docs("dspy.evaluate", Path("docs/api/evaluate"))
    generate_module_docs("dspy.predict", Path("docs/api/predict"))
    generate_module_docs("dspy.primitives", Path("docs/api/primitives"))
    generate_module_docs("dspy.signatures", Path("docs/api/signatures"))
    generate_module_docs("dspy.teleprompt", Path("docs/api/teleprompt"))
