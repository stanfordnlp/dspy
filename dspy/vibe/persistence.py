from __future__ import annotations

import textwrap
from typing import NamedTuple

_SIG_HASH_PREFIX = "# __VIBE_SIGNATURE_HASH__:"
_PREDICTORS_BEGIN = "# __VIBE_PREDICTORS_BEGIN__"
_PREDICTORS_END = "# __VIBE_PREDICTORS_END__"
_FORWARD_BEGIN = "# __VIBE_FORWARD_BEGIN__"
_FORWARD_END = "# __VIBE_FORWARD_END__"
_CLASS_INDENT = "    "

_FILE_HEADER_DOCSTRING = '''"""Implementation of a dspy.Vibe module — managed automatically.

This file starts life as a baseline that delegates to dspy.RLM, and is rewritten in place
by dspy.GEPA when you optimize the module (decomposing the task into focused predictors and
plain Python). It is meant to be readable and reviewable.

- You may edit the PREDICTORS dict and the forward() body between the marker comments below;
  on the next run dspy.Vibe parses those regions back out and runs your code as-is.
- __VIBE_SIGNATURE_HASH__ guards against stale code: if you change the Signature, the hash
  no longer matches and dspy.Vibe regenerates the baseline for the new Signature (re-run
  dspy.GEPA to re-optimize).

Leave the marker comments and the signature-hash line intact.
"""
'''


class PersistedVibe(NamedTuple):
    """Parsed view of a persisted Vibe module file."""

    signature_hash: str
    predictors_src: str
    forward_src: str


def render_persisted_file(
    *,
    signature_hash: str,
    signature_name: str,
    predictors_src: str,
    forward_src: str,
) -> str:
    """Render a complete, self-contained Vibe module file."""
    module_name = f"{signature_name}Module"
    predictors_block = _wrap_with_markers(predictors_src, _PREDICTORS_BEGIN, _PREDICTORS_END)
    forward_block = _wrap_with_markers(forward_src, _FORWARD_BEGIN, _FORWARD_END)

    return (
        _FILE_HEADER_DOCSTRING
        + f"\n{_SIG_HASH_PREFIX} {signature_hash}\n\n"
        + "import dspy\n\n\n"
        + f"class {module_name}(dspy.Module):\n"
        + f'    """Standalone dspy.Module compiled from the {signature_name} signature."""\n'
        + "\n"
        + predictors_block
        + "\n\n"
        + "    def __init__(self) -> None:\n"
        + "        super().__init__()\n"
        + "        for _name, _predictor in self.PREDICTORS.items():\n"
        + "            setattr(self, _name, _predictor)\n"
        + "\n"
        + forward_block
        + "\n"
    )


def parse_persisted_file(text: str) -> PersistedVibe | None:
    """Extract the signature hash and source regions from a Vibe file."""
    try:
        signature_hash = text.split(_SIG_HASH_PREFIX, 1)[1].splitlines()[0].strip()
        predictors_raw = text.split(_PREDICTORS_BEGIN, 1)[1].split(_PREDICTORS_END, 1)[0]
        forward_raw = text.split(_FORWARD_BEGIN, 1)[1].split(_FORWARD_END, 1)[0]
    except IndexError:
        return None

    return PersistedVibe(
        signature_hash=signature_hash,
        predictors_src=textwrap.dedent(predictors_raw).strip("\n"),
        forward_src=textwrap.dedent(forward_raw).strip("\n"),
    )


def _wrap_with_markers(src: str, begin: str, end: str) -> str:
    """Wrap a module-scope source string with markers, indented for class scope."""
    return textwrap.indent(f"{begin}\n{src.strip()}\n{end}", _CLASS_INDENT)
