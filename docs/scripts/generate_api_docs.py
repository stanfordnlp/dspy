from pathlib import Path
import importlib
import inspect
import pkgutil
import shutil


def get_module_contents(module):
    """Get all public classes and functions from a module."""
    if hasattr(module, "__all__"):
        return module.__all__

    contents = []
    for name, obj in inspect.getmembers(module):
        if (
            (inspect.isclass(obj) or inspect.isfunction(obj))
            and obj.__module__.startswith(module.__name__)
            and not name.startswith("_")
        ):
            contents.append(name)
    return contents


def generate_module_docs(module_path: str, output_dir: Path):
    """Generate API documentation for a module and its submodules."""
    # Import the module
    module = importlib.import_module(module_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all public classes and functions
    contents = get_module_contents(module)

    # Create a single page for the module
    module_name = module_path.split(".")[-1]
    members_list = "\n          - ".join(contents)

    page_content = f"""# {module_name.title()} API Reference

::: {module_path}
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        members:
          - {members_list}
        docstring_section_style: google
        show_root_full_path: false
        show_object_full_path: false
        separate_signature: false
"""
    with open(output_dir / f"{module_name}.md", "w") as f:
        f.write(page_content)

    if hasattr(module, "__path__"):
        # Handle submodules
        for submodule in pkgutil.iter_modules(module.__path__, prefix=f"{module.__name__}."):
            submodule_name = submodule.name.split(".")[-1]
            submodule_path = Path(submodule_name)
            generate_module_docs(submodule.name, output_dir / submodule_path)


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
