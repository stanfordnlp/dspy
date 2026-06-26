from __future__ import annotations

from typing import NamedTuple

_SIG_HASH_PREFIX = "# __VIBE_SIGNATURE_HASH__:"
_SIG_SPEC_BEGIN = "# __VIBE_SIGNATURE_BEGIN__"
_SIG_SPEC_END = "# __VIBE_SIGNATURE_END__"
_MODULE_BEGIN = "# __VIBE_MODULE_BEGIN__"
_MODULE_END = "# __VIBE_MODULE_END__"

# ASCII-only header (no em-dashes etc.): the file is written as UTF-8 but is also read and
# hand-edited by tools on platforms whose default text encoding is not UTF-8 (e.g. Windows
# cp1252), so we keep the bytes portable.
_FILE_HEADER_DOCSTRING = '''"""Implementation of a dspy.Vibe module -- managed automatically.

This file starts life as a baseline that delegates to dspy.RLM, and is rewritten in place
by dspy.GEPA when you optimize the module (decomposing the task into focused predictors and
plain Python). It is a normal, runnable dspy.Module and is meant to be readable and editable.

- You may edit the module class between the __VIBE_MODULE_BEGIN__/__VIBE_MODULE_END__ markers;
  on the next run dspy.Vibe parses that class back out and runs your code as-is.
- __VIBE_SIGNATURE__ records the Signature this module was vibed from (for you and for GEPA).
- __VIBE_SIGNATURE_HASH__ guards against stale code: if you change the Signature, the hash no
  longer matches and dspy.Vibe regenerates the baseline for the new Signature (re-run dspy.GEPA
  to re-optimize).

Leave the marker comments and the signature-hash line intact.
"""
'''


class PersistedVibe(NamedTuple):
    """Parsed view of a persisted Vibe module file."""

    signature_hash: str
    module_src: str


def render_persisted_file(
    *,
    signature_hash: str,
    signature_name: str,
    module_src: str,
    signature_spec: str,
) -> str:
    """Render a complete, self-contained Vibe module file.

    ``module_src`` is the full ``dspy.Module`` subclass source (top-level, no indentation).
    ``signature_spec`` is the human-readable rendering of the Signature this module was vibed
    from; it is embedded as a comment block so the file documents what it implements (and so
    the spec survives even after GEPA decomposes the top-level signature away).
    """
    signature_block = "\n".join(f"# {line}" for line in signature_spec.strip().splitlines())
    return (
        _FILE_HEADER_DOCSTRING
        + f"\n{_SIG_HASH_PREFIX} {signature_hash}\n\n"
        + f"{_SIG_SPEC_BEGIN}\n{signature_block}\n{_SIG_SPEC_END}\n\n"
        + "import dspy\n\n\n"
        + f"{_MODULE_BEGIN}\n"
        + module_src.strip()
        + f"\n{_MODULE_END}\n"
    )


def parse_persisted_file(text: str) -> PersistedVibe | None:
    """Extract the signature hash and the module-class source from a Vibe file."""
    try:
        signature_hash = text.split(_SIG_HASH_PREFIX, 1)[1].splitlines()[0].strip()
        module_raw = text.split(_MODULE_BEGIN, 1)[1].split(_MODULE_END, 1)[0]
    except IndexError:
        return None

    return PersistedVibe(signature_hash=signature_hash, module_src=module_raw.strip("\n"))
