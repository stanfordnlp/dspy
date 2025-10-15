import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

import dspy

API_MAPPING = {
    "models": [
        dspy.LM,
        dspy.Embedder,
    ],
    "primitives": [
        dspy.Audio,
        dspy.Code,
        dspy.Example,
        dspy.Image,
        dspy.History,
        dspy.Prediction,
        dspy.Tool,
        dspy.ToolCalls,
    ],
    "signatures": [
        dspy.Signature,
        dspy.InputField,
        dspy.OutputField,
    ],
    "adapters": [
        dspy.Adapter,
        dspy.ChatAdapter,
        dspy.JSONAdapter,
        dspy.TwoStepAdapter,
    ],
    "modules": [
        dspy.Module,
        dspy.Predict,
        dspy.ChainOfThought,
        dspy.ReAct,
        dspy.ProgramOfThought,
        dspy.MultiChainComparison,
        dspy.Parallel,
        dspy.BestOfN,
        dspy.Refine,
    ],
    "tools": [
        dspy.ColBERTv2,
        dspy.retrievers.Embeddings,
        dspy.PythonInterpreter,
    ],
    "utils": [
        dspy.inspect_history,
        dspy.load,
        dspy.asyncify,
        dspy.streamify,
        dspy.enable_logging,
        dspy.disable_logging,
        dspy.enable_litellm_logging,
        dspy.disable_litellm_logging,
        dspy.configure_cache,
        dspy.streaming.StatusMessageProvider,
        dspy.streaming.StatusMessage,
        dspy.streaming.StreamListener,
    ],
    "evaluation": [
        dspy.Evaluate,
        dspy.evaluate.answer_exact_match,
        dspy.evaluate.answer_passage_match,
        dspy.evaluate.SemanticF1,
        dspy.evaluate.CompleteAndGrounded,
        dspy.evaluate.EvaluationResult,
    ],
    "optimizers": [
        dspy.LabeledFewShot,
        dspy.BootstrapFewShot,
        dspy.BootstrapFewShotWithRandomSearch,
        dspy.MIPROv2,
        dspy.BetterTogether,
        dspy.BootstrapFinetune,
        dspy.COPRO,
        dspy.Ensemble,
        dspy.KNN,
        dspy.KNNFewShot,
        dspy.InferRules,
        dspy.GEPA,
    ],
    "experimental": [
        dspy.experimental.Citations,
        dspy.experimental.Document,
    ],
}

LOCATION_OVERRIDES = {
    "docs/api/optimizers/GEPA.md": "docs/api/optimizers/GEPA/overview.md",
}

def should_document_method(obj):
    name = obj.__name__
    # Exclude methods not defined in dspy, such as `model_dump_json` from pydantic.
    module = getattr(obj, "__module__", "")
    if not module or not module.startswith("dspy"):
        return False
    # Exclude private and dunder methods, but include `__call__`
    if name == "__call__" or not name.startswith("_"):
        return True
    return False


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
            (inspect.isclass(obj) or (inspect.isroutine(obj) and should_document_method(obj)))
            # classes or functions in experimental module are not located in dspy/experimental
            and (obj.__module__.startswith(module.__name__) or module.__name__.startswith("dspy.experimental"))
            and not name.startswith("_")
        ):
            contents[name] = obj
    return contents


def get_public_methods(cls):
    """Returns a list of all public methods in a class."""
    return [
        name
        for name, member in inspect.getmembers(
            cls, predicate=lambda x: inspect.isroutine(x) and should_document_method(x)
        )
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

    # We need to put ::: at last to avoid unclosed div. See https://github.com/danielfrg/mkdocs-jupyter/issues/231 for more details.
    return f"""<!-- START_API_REF -->
::: {module_path}.{name}
    handler: python
    options:{members_config}
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->
"""


def get_api_category(obj):
    for category, objects in API_MAPPING.items():
        if obj in objects:
            return category
    return None


def read_existing_content(file_path: Path) -> tuple[str, str]:
    """Read existing file content and split into pre and post API reference sections.

    Returns:
        tuple[str, str]: (content_before_api_ref, content_after_api_ref)
        If file doesn't exist or no API ref section found, returns empty strings.
    """
    if not file_path.exists():
        return "", ""

    content = file_path.read_text()

    # Look for our specific API reference markers
    api_start_marker = "<!-- START_API_REF -->"
    api_end_marker = "<!-- END_API_REF -->"

    api_start = content.find(api_start_marker)
    if api_start == -1:
        # No API section found, treat all content as pre-content
        return content, ""

    api_end = content.find(api_end_marker)
    if api_end == -1:
        # Start marker found but no end marker - treat rest of file as post-content
        api_end = len(content)
    else:
        api_end = api_end + len(api_end_marker)

    return content[:api_start].rstrip(), content[api_end:].lstrip()


def write_doc_file(file_path: Path, title: str, api_content: str):
    """Write documentation to file while preserving existing content."""
    pre_content, post_content = read_existing_content(file_path)

    # If no pre-content exists, add the title
    if not pre_content:
        pre_content = f"# {title}\n"

    # Combine all sections
    full_content = f"{pre_content}\n\n{api_content}\n{post_content}".strip() + "\n"

    # Write the combined content
    file_path.write_text(full_content)


def generate_md_docs(output_dir: Path, excluded_modules=None):
    """Generate documentation for all public classes and functions in the dspy package.

    Args:
        output_dir: The directory to write the documentation to, e.g. "docs/api"
        excluded_modules: A list of modules to exclude from documentation, e.g. ["dspy.dsp"]
    """
    module = importlib.import_module("dspy")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_contents = get_module_contents(module)
    objects_processed = {}

    # Generate docs for root-level objects, e.g. dspy.Predict, dspy.Example, etc.
    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            continue

        category = get_api_category(obj)
        if category is None:
            # Skip if the object is not in the API mapping.
            continue

        page_content = generate_doc_page(name, "dspy", obj, is_root=True)
        file_path = output_dir / category / f"{name}.md"
        if file_path.as_posix() in LOCATION_OVERRIDES:
            file_path = Path(LOCATION_OVERRIDES[file_path.as_posix()])
        write_doc_file(file_path, f"dspy.{name}", page_content)

        objects_processed[f"{obj.__module__}.{name}"] = obj

    for submodule in pkgutil.iter_modules(module.__path__, prefix=f"{module.__name__}."):
        submodule_name = submodule.name.split(".")[-1]

        # Skip if this is a private module or not in __init__.py
        if submodule_name.startswith("_") or submodule_name not in init_contents:
            continue

        generate_md_docs_submodule(submodule.name, output_dir, objects_processed, excluded_modules)


def generate_md_docs_submodule(module_path: str, output_dir: Path, objects_processed=None, excluded_modules=None):
    """Recursively generate documentation for a submodule.

    We generate docs for all public classes and functions in the submodule, then recursively generate docs for all
    submodules within the submodule.

    Args:
        module_path: The path to the submodule, e.g. "dspy.predict"
        output_dir: The directory to write the documentation to, e.g. "docs/api/predict"
        objects_processed: A dictionary of objects that have already been processed, used to avoid redundant processing.
        excluded_modules: A list of modules to exclude from documentation, e.g. ["dspy.dsp"]
    """
    if excluded_modules and module_path in excluded_modules:
        return

    try:
        module = importlib.import_module(module_path)
    except ImportError:
        print(f"Skipping {module_path} due to import error")
        return

    init_contents = get_module_contents(module)

    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            continue

        category = get_api_category(obj)
        if category is None:
            # Skip if the object is not in the API mapping.
            continue

        full_name = f"{obj.__module__}.{name}"
        if full_name not in objects_processed:
            # Only generate docs for objects that are not root-level objects.
            page_content = generate_doc_page(name, module_path, obj, is_root=False)
            file_path = output_dir / category / f"{name}.md"
            if file_path.as_posix() in LOCATION_OVERRIDES:
                file_path = Path(LOCATION_OVERRIDES[file_path.as_posix()])
            write_doc_file(file_path, f"{module_path}.{name}", page_content)

            objects_processed[full_name] = obj

    for name, obj in init_contents.items():
        if inspect.ismodule(obj):
            generate_md_docs_submodule(f"{module_path}.{name}", output_dir / name, objects_processed)


def remove_empty_dirs(path: Path):
    """Recursively remove empty directories."""
    for child in path.glob("*"):
        if child.is_dir():
            remove_empty_dirs(child)

    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


if __name__ == "__main__":
    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    # Create category directories if they don't exist
    for category in API_MAPPING.keys():
        subpath = api_dir / category
        subpath.mkdir(parents=True, exist_ok=True)

    excluded_modules = ["dspy.dsp"]
    generate_md_docs(api_dir, excluded_modules=excluded_modules)

    # Clean up empty directories
    remove_empty_dirs(api_dir)
