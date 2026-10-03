"""
SkillsReAct module for DSPy.

A ReAct agent that integrates skill-based capabilities following the Agent Skills
specification (agentskills.io). Skills are discovered from directories containing
SKILL.md files with YAML frontmatter and markdown instructions.

Overview
--------
This module provides progressive disclosure of skill information:

1. **Discovery**: Load only metadata (name, description) for all skills
2. **Activation**: Load full instructions when a skill is needed
3. **Resources**: Access references/assets on demand
4. **Scripts**: Run sandboxed Python scripts from skills (opt-in)

Quick Start
-----------
Basic usage without script execution::

    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    agent = dspy.SkillsReAct(
        signature="request: str -> response: str",
        skill_dirs=["./skills"],
        max_iters=10,
    )

    result = agent(request="Help me format this JSON file")
    print(result.response)

With script execution enabled (unsafe mode for development)::

    agent = dspy.SkillsReAct(
        signature="request: str -> response: str",
        skill_dirs=["./skills"],
        max_iters=10,
        enable_scripts=True,
        sandbox_mode="unsafe",  # For local dev only
    )

With WASM sandbox (secure, requires Deno)::

    agent = dspy.SkillsReAct(
        signature="request: str -> response: str",
        skill_dirs=["./skills"],
        enable_scripts=True,
        sandbox_mode="wasm",  # Secure sandbox, requires Deno
    )

Constructor Arguments
---------------------
signature : str or Signature
    The DSPy signature defining input/output fields.
    Example: "request: str -> response: str"

skill_dirs : list[str | Path]
    List of directories to scan for skills. Each directory should contain
    subdirectories with SKILL.md files.
    Example: ["./skills", "~/.skills"]

tools : list[Callable], optional
    Additional tools to provide to the agent alongside the skill meta-tools.

max_iters : int, default=20
    Maximum number of ReAct iterations before stopping.

enable_scripts : bool, default=False
    Whether to enable script execution. When True, adds the `run_skill_script`
    tool allowing the agent to execute Python scripts from skills.
    **Disabled by default for security.**

script_timeout : int, default=30
    Timeout in seconds for script execution.

enable_network : bool, default=False
    Whether scripts can access the network (only applies to WASM mode).

sandbox_mode : "wasm" | "unsafe", default="unsafe"
    Sandboxing backend for script execution:

    - **"wasm"**: Uses DSPy's PythonInterpreter with Deno/Pyodide for true
      isolation. Most secure, but requires Deno to be installed. Only supports
      pure Python scripts with Pyodide-compatible dependencies.

    - **"unsafe"**: Direct subprocess execution with no sandboxing.
      **Use only for local development/testing with trusted skills.**

Available Tools
---------------
The agent automatically has access to these skill meta-tools:

**list_skills()**
    Returns a formatted list of all available skills with names and descriptions.
    Use this to discover what capabilities are available.

**activate_skill(skill_name: str)**
    Activates a skill by loading its full instructions.
    Must be called before using a skill's resources or scripts.

**read_skill_resource(skill_name: str, resource_type: str, filename: str)**
    Reads a reference or asset file from an activated skill.
    - resource_type: "references" or "assets"
    - Returns file contents as text

**run_skill_script(skill_name: str, script_name: str, arguments: str)** *(if enable_scripts=True)*
    Runs a Python script from an activated skill.
    - script_name: e.g., "process_data.py"
    - arguments: space-separated args, e.g., "--input file.txt --format json"

**bash(command: str)** *(auto-created if skills declare allowed-tools)*
    Executes bash commands allowed by the active skill's `allowed-tools` declaration.
    Only available in "unsafe" sandbox mode.

SKILL.md Format
---------------
Each skill is defined by a SKILL.md file with YAML frontmatter::

    ---
    name: my-skill
    description: Short description of what this skill does
    license: MIT                    # optional
    compatibility: Python 3.9+      # optional
    allowed-tools: Bash(gog:*)      # optional, enables bash tool for specific commands
    metadata:                       # optional key-value pairs
      author: Your Name
      version: "1.0"
    permissions:                    # optional, for script execution
      read_paths: ["/tmp"]
      write_paths: []
      env_vars: ["HOME", "PATH"]
      network: false
      timeout: 60
    ---

    # My Skill

    Detailed instructions for how and when to use this skill.
    This markdown body is loaded when the skill is activated.

    ## Usage

    Describe usage patterns, examples, etc.

Skill Directory Structure
-------------------------
::

    skills/
    ├── my-skill/
    │   ├── SKILL.md              # Required: skill definition
    │   ├── scripts/              # Optional: executable Python scripts
    │   │   ├── process.py
    │   │   └── validate.py
    │   ├── references/           # Optional: reference documentation
    │   │   └── api-docs.md
    │   └── assets/               # Optional: static files
    │       └── templates/
    │           └── config.json
    └── another-skill/
        └── SKILL.md

Sandbox Modes
-------------
**WASM Mode ("wasm")**:
    - True isolation via Deno/Pyodide WASM runtime
    - No access to host filesystem (scripts run in isolated environment)
    - No access to system files like /proc, /sys
    - Scripts that need subprocess won't work
    - Only pure Python with Pyodide-compatible packages
    - Requires Deno: https://docs.deno.com/runtime/getting_started/installation/

**Unsafe Mode ("unsafe")**:
    - Direct subprocess execution
    - Scripts have access to the host system
    - **WARNING**: Use only for development/testing with trusted skills
    - A warning is logged on first script execution

Exceptions
----------
SkillError
    Base exception for all skill-related errors.

ParseError
    Raised when SKILL.md parsing fails (missing frontmatter, invalid YAML).

ValidationError
    Raised when skill validation fails (missing required fields).

SkillNotFoundError
    Raised when a requested skill doesn't exist.

ResourceNotFoundError
    Raised when a skill resource (reference/asset/script) doesn't exist.

ExecutionError
    Raised when script execution fails.

SecurityError
    Raised when a security violation is detected (path traversal, etc.).

Examples
--------
Using with multiple skill directories::

    agent = dspy.SkillsReAct(
        signature="task: str -> result: str",
        skill_dirs=[
            Path.home() / ".skills",     # User's personal skills
            Path("./project-skills"),     # Project-specific skills
        ],
        max_iters=15,
    )

Combining with custom tools::

    def search_web(query: str) -> str:
        '''Search the web for information.'''
        # ... implementation
        return results

    agent = dspy.SkillsReAct(
        signature="question: str -> answer: str",
        skill_dirs=["./skills"],
        tools=[search_web],  # Custom tools alongside skill tools
    )

Accessing the skill manager directly::

    agent = dspy.SkillsReAct(
        signature="request: str -> response: str",
        skill_dirs=["./skills"],
    )

    # List discovered skills
    for skill in agent.skill_manager.list_skills():
        print(f"{skill.name}: {skill.description}")

    # Manually activate a skill
    skill = agent.skill_manager.activate("my-skill")
    print(skill.instructions)

See Also
--------
- Agent Skills Specification: https://agentskills.io
- DSPy ReAct: dspy.predict.react.ReAct
- Deno Installation: https://docs.deno.com/runtime/getting_started/installation/
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

import yaml

from dspy.adapters.types.tool import Tool
from dspy.predict.react import ReAct
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

SandboxMode = Literal["wasm", "unsafe"]


# =============================================================================
# Exceptions
# =============================================================================


class SkillError(Exception):
    """Base exception for all skill-related errors."""

    pass


class ParseError(SkillError):
    """Raised when SKILL.md parsing fails."""

    pass


class ValidationError(SkillError):
    """Raised when skill validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or [message]


class SkillNotFoundError(SkillError):
    """Raised when a requested skill doesn't exist."""

    def __init__(self, skill_name: str, available: list[str] | None = None):
        self.skill_name = skill_name
        self.available = available or []
        msg = f"Skill '{skill_name}' not found"
        if available:
            msg += f". Available: {', '.join(available)}"
        super().__init__(msg)


class ResourceNotFoundError(SkillError):
    """Raised when a skill resource (reference/asset) doesn't exist."""

    def __init__(self, skill_name: str, resource_type: str, filename: str):
        self.skill_name = skill_name
        self.resource_type = resource_type
        self.filename = filename
        super().__init__(f"Resource '{filename}' not found in {skill_name}/{resource_type}/")


class ExecutionError(SkillError):
    """Raised when script execution fails."""

    def __init__(self, script_path: Path, reason: str, stderr: str | None = None):
        self.script_path = script_path
        self.reason = reason
        self.stderr = stderr
        super().__init__(f"Execution of {script_path} failed: {reason}")


class SecurityError(SkillError):
    """Raised when a security violation is detected."""

    pass


# =============================================================================
# Data Models
# =============================================================================


class SkillState(Enum):
    """Represents the loading state of a skill."""

    DISCOVERED = "discovered"  # Metadata loaded only
    ACTIVATED = "activated"  # Full instructions loaded


@dataclass
class SkillPermissions:
    """Permissions for a skill's script execution.

    These can be defined in SKILL.md frontmatter under the 'permissions' key
    and are merged with module-level settings (more restrictive wins).
    """

    read_paths: list[str] = field(default_factory=list)
    write_paths: list[str] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    network: bool = False
    timeout: int = 30


@dataclass
class ExecutionResult:
    """Result of script execution."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class Skill:
    """Represents a skill with its current load state.

    Attributes:
        name: The skill name (from frontmatter)
        description: What the skill does and when to use it
        path: Path to the skill directory
        state: Current loading state (DISCOVERED or ACTIVATED)
        license: Optional license information
        compatibility: Optional environment requirements
        allowed_tools: Optional space-delimited list of pre-approved tools (e.g., "Bash(gog:*)")
        metadata: Optional key-value metadata mapping
        permissions: Optional script execution permissions
        instructions: Full skill instructions (loaded on activation)
    """

    name: str
    description: str
    path: Path
    state: SkillState = SkillState.DISCOVERED
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    permissions: SkillPermissions | None = None
    instructions: str | None = None

    @property
    def scripts_dir(self) -> Path | None:
        """Return scripts/ directory if it exists."""
        scripts = self.path / "scripts"
        return scripts if scripts.is_dir() else None

    @property
    def references_dir(self) -> Path | None:
        """Return references/ directory if it exists."""
        refs = self.path / "references"
        return refs if refs.is_dir() else None

    @property
    def assets_dir(self) -> Path | None:
        """Return assets/ directory if it exists."""
        assets = self.path / "assets"
        return assets if assets.is_dir() else None

    def has_scripts(self) -> bool:
        """Check if the skill has a scripts directory."""
        return self.scripts_dir is not None

    def has_references(self) -> bool:
        """Check if the skill has a references directory."""
        return self.references_dir is not None

    def has_assets(self) -> bool:
        """Check if the skill has an assets directory."""
        return self.assets_dir is not None


# =============================================================================
# SKILL.md Parser
# =============================================================================


def find_skill_md(skill_dir: Path) -> Path | None:
    """Find the SKILL.md file in a skill directory.

    Prefers SKILL.md (uppercase) but accepts skill.md (lowercase).

    Args:
        skill_dir: Path to the skill directory

    Returns:
        Path to the SKILL.md file, or None if not found
    """
    for name in ("SKILL.md", "skill.md"):
        path = skill_dir / name
        if path.exists():
            return path
    return None


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
        content: Raw content of SKILL.md file

    Returns:
        Tuple of (metadata dict, markdown body)

    Raises:
        ParseError: If frontmatter is missing or invalid
    """
    if not content.startswith("---"):
        raise ParseError("SKILL.md must start with YAML frontmatter (---)")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ParseError("SKILL.md frontmatter not properly closed with ---")

    frontmatter_str = parts[1]
    body = parts[2].strip()

    try:
        metadata = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise ParseError(f"Invalid YAML in frontmatter: {e}")

    if not isinstance(metadata, dict):
        raise ParseError("SKILL.md frontmatter must be a YAML mapping")

    # Convert metadata values to strings
    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
        metadata["metadata"] = {str(k): str(v) for k, v in metadata["metadata"].items()}

    return metadata, body


def read_skill(skill_dir: Path, load_instructions: bool = False) -> Skill:
    """Read a skill from a directory.

    Args:
        skill_dir: Path to the skill directory
        load_instructions: If True, load full instructions (ACTIVATED state)

    Returns:
        Skill with parsed metadata and optionally instructions

    Raises:
        ParseError: If SKILL.md is missing or has invalid YAML
        ValidationError: If required fields (name, description) are missing
    """
    skill_dir = Path(skill_dir).resolve()
    skill_md = find_skill_md(skill_dir)

    if skill_md is None:
        raise ParseError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text()
    metadata, body = parse_frontmatter(content)

    # Validate required fields
    if "name" not in metadata:
        raise ValidationError("Missing required field in frontmatter: name")
    if "description" not in metadata:
        raise ValidationError("Missing required field in frontmatter: description")

    name = metadata["name"]
    description = metadata["description"]

    if not isinstance(name, str) or not name.strip():
        raise ValidationError("Field 'name' must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValidationError("Field 'description' must be a non-empty string")

    state = SkillState.ACTIVATED if load_instructions else SkillState.DISCOVERED
    instructions = body if load_instructions else None

    # Parse permissions if present
    permissions = None
    if "permissions" in metadata and isinstance(metadata["permissions"], dict):
        perm_data = metadata["permissions"]
        permissions = SkillPermissions(
            read_paths=perm_data.get("read_paths", []),
            write_paths=perm_data.get("write_paths", []),
            env_vars=perm_data.get("env_vars", []),
            network=perm_data.get("network", False),
            timeout=perm_data.get("timeout", 30),
        )

    return Skill(
        name=name.strip(),
        description=description.strip(),
        path=skill_dir,
        state=state,
        license=metadata.get("license"),
        compatibility=metadata.get("compatibility"),
        allowed_tools=metadata.get("allowed-tools"),
        metadata=metadata.get("metadata", {}),
        permissions=permissions,
        instructions=instructions,
    )


def read_instructions(skill_dir: Path) -> str:
    """Read just the instructions (body) from a SKILL.md file.

    Args:
        skill_dir: Path to the skill directory

    Returns:
        The markdown body of the SKILL.md file

    Raises:
        ParseError: If SKILL.md is missing or has invalid format
    """
    skill_dir = Path(skill_dir).resolve()
    skill_md = find_skill_md(skill_dir)

    if skill_md is None:
        raise ParseError(f"SKILL.md not found in {skill_dir}")

    content = skill_md.read_text()
    _, body = parse_frontmatter(content)
    return body


# =============================================================================
# Skill Manager
# =============================================================================


class SkillManager:
    """Manages skill discovery, loading, and state.

    Implements progressive disclosure:
    1. Discovery: Load only metadata (name, description) for all skills
    2. Activation: Load full instructions when a skill is needed
    3. Resources: Access references/assets on demand

    Example:
        >>> manager = SkillManager([Path("~/.skills").expanduser()])
        >>> manager.discover()
        ['pdf', 'docx', 'webapp-testing']
        >>> skill = manager.activate("pdf")
        >>> print(skill.instructions[:100])
        # PDF Processing...
    """

    def __init__(self, skill_dirs: list[Path]):
        """Initialize the SkillManager.

        Args:
            skill_dirs: List of directories to scan for skills
        """
        self._skill_dirs = [Path(d).expanduser().resolve() for d in skill_dirs]
        self._skills: dict[str, Skill] = {}

    def discover(self) -> list[str]:
        """Scan all configured directories for valid skills.

        Loads metadata only (progressive disclosure level 1).

        Returns:
            List of discovered skill names
        """
        self._skills.clear()
        discovered = []

        for skill_dir in self._skill_dirs:
            if not skill_dir.exists():
                logger.warning(f"Skill directory does not exist: {skill_dir}")
                continue

            if not skill_dir.is_dir():
                logger.warning(f"Skill path is not a directory: {skill_dir}")
                continue

            # Find all subdirectories that contain SKILL.md
            for subdir in skill_dir.iterdir():
                if not subdir.is_dir():
                    continue

                skill_md = find_skill_md(subdir)
                if skill_md is None:
                    continue

                # Load metadata only
                try:
                    skill = read_skill(subdir, load_instructions=False)
                    if skill.name in self._skills:
                        logger.warning(
                            f"Duplicate skill name '{skill.name}' at {subdir}, "
                            f"keeping first from {self._skills[skill.name].path}"
                        )
                        continue
                    self._skills[skill.name] = skill
                    discovered.append(skill.name)
                except Exception as e:
                    logger.warning(f"Failed to load skill from {subdir}: {e}")
                    continue

        logger.info(f"Discovered {len(discovered)} skills: {discovered}")
        return discovered

    def list_skills(self) -> list[Skill]:
        """Return all discovered skills with their metadata.

        Returns:
            List of Skill objects (metadata only, unless activated)
        """
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: The skill name

        Returns:
            The Skill if found, None otherwise
        """
        return self._skills.get(name)

    def activate(self, name: str) -> Skill:
        """Activate a skill by loading its full instructions.

        Implements progressive disclosure level 2.

        Args:
            name: Skill name to activate

        Returns:
            The activated Skill with instructions populated

        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(name, list(self._skills.keys()))

        if skill.state == SkillState.ACTIVATED:
            # Already activated
            return skill

        # Load full instructions
        try:
            instructions = read_instructions(skill.path)
            skill.instructions = instructions
            skill.state = SkillState.ACTIVATED
            logger.info(f"Activated skill: {name}")
            return skill
        except Exception as e:
            raise ValidationError(f"Failed to load instructions for skill '{name}': {e}")

    def list_scripts(self, skill_name: str) -> list[str]:
        """List available script files for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of script filenames (only .py files)

        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self._skills.keys()))

        scripts_dir = skill.scripts_dir
        if scripts_dir is None:
            return []

        return [f.name for f in scripts_dir.iterdir() if f.is_file() and f.suffix == ".py"]

    def list_references(self, skill_name: str) -> list[str]:
        """List available reference files for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of reference filenames

        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self._skills.keys()))

        refs_dir = skill.references_dir
        if refs_dir is None:
            return []

        return [f.name for f in refs_dir.iterdir() if f.is_file()]

    def list_assets(self, skill_name: str) -> list[str]:
        """List available asset files for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of asset filenames

        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self._skills.keys()))

        assets_dir = skill.assets_dir
        if assets_dir is None:
            return []

        # For assets, we might want to recurse into subdirectories
        assets = []
        for item in assets_dir.rglob("*"):
            if item.is_file():
                # Return relative path from assets_dir
                assets.append(str(item.relative_to(assets_dir)))
        return assets

    def get_resource_path(self, skill_name: str, resource_type: str, filename: str) -> Path:
        """Get the full path to a skill resource.

        Args:
            skill_name: Name of the skill
            resource_type: One of 'references', 'assets'
            filename: Name of the file (can include subdirectory for assets)

        Returns:
            Absolute path to the resource

        Raises:
            SkillNotFoundError: If skill doesn't exist
            ResourceNotFoundError: If resource doesn't exist
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self._skills.keys()))

        if resource_type == "references":
            base_dir = skill.references_dir
        elif resource_type == "assets":
            base_dir = skill.assets_dir
        else:
            raise ValueError(f"Invalid resource_type '{resource_type}'. Must be 'references' or 'assets'.")

        if base_dir is None:
            raise ResourceNotFoundError(skill_name, resource_type, filename)

        resource_path = base_dir / filename

        # Security: ensure path doesn't escape the resource directory
        try:
            resource_path = resource_path.resolve()
            base_dir_resolved = base_dir.resolve()
            if not str(resource_path).startswith(str(base_dir_resolved)):
                raise ResourceNotFoundError(skill_name, resource_type, filename)
        except Exception:
            raise ResourceNotFoundError(skill_name, resource_type, filename)

        if not resource_path.exists():
            raise ResourceNotFoundError(skill_name, resource_type, filename)

        return resource_path

    def get_script_path(self, skill_name: str, script_name: str) -> Path:
        """Get the full path to a skill script.

        Args:
            skill_name: Name of the skill
            script_name: Name of the script file (must be .py)

        Returns:
            Absolute path to the script

        Raises:
            SkillNotFoundError: If skill doesn't exist
            ResourceNotFoundError: If script doesn't exist
            SecurityError: If script name is invalid
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self._skills.keys()))

        scripts_dir = skill.scripts_dir
        if scripts_dir is None:
            raise ResourceNotFoundError(skill_name, "scripts", script_name)

        # Security: only allow .py files
        if not script_name.endswith(".py"):
            raise SecurityError(f"Only Python scripts (.py) are allowed, got: {script_name}")

        script_path = scripts_dir / script_name

        # Security: ensure path doesn't escape the scripts directory
        try:
            script_path = script_path.resolve()
            scripts_dir_resolved = scripts_dir.resolve()
            if not str(script_path).startswith(str(scripts_dir_resolved)):
                raise SecurityError(f"Script path escapes skill directory: {script_name}")
        except SecurityError:
            raise
        except Exception:
            raise ResourceNotFoundError(skill_name, "scripts", script_name)

        if not script_path.exists():
            raise ResourceNotFoundError(skill_name, "scripts", script_name)

        return script_path


# =============================================================================
# Script Executors
# =============================================================================


class BaseScriptExecutor(ABC):
    """Abstract base class for script executors."""

    def __init__(
        self,
        timeout: int = 30,
        allow_network: bool = False,
    ):
        self.timeout = timeout
        self.allow_network = allow_network

    @abstractmethod
    def run(
        self,
        script_path: Path,
        arguments: list[str],
        working_dir: Path,
        timeout: int | None = None,
        permissions: SkillPermissions | None = None,
    ) -> ExecutionResult:
        """Execute a script.

        Args:
            script_path: Path to the script file
            arguments: Command-line arguments
            working_dir: Working directory for execution
            timeout: Override default timeout
            permissions: Optional skill-specific permissions

        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        pass

    def _validate_script_path(self, script_path: Path, skill_dir: Path) -> None:
        """Validate that the script path is within the skill directory."""
        try:
            script_resolved = script_path.resolve()
            skill_resolved = skill_dir.resolve()

            if not str(script_resolved).startswith(str(skill_resolved) + os.sep):
                if script_resolved != skill_resolved:
                    raise SecurityError(f"Script path '{script_path}' is outside skill directory")
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Invalid script path: {e}")


class UnsafeScriptExecutor(BaseScriptExecutor):
    """Script executor using direct subprocess execution.

    WARNING: This provides NO sandboxing - use only for local development/testing.
    A warning is logged on first use.
    """

    _warned = False

    def __init__(
        self,
        timeout: int = 30,
        allow_network: bool = False,
    ):
        super().__init__(timeout, allow_network)

    def _warn_once(self):
        if not UnsafeScriptExecutor._warned:
            logger.warning(
                "Using 'unsafe' sandbox mode - scripts run with NO SANDBOXING. "
                "Only use this for local development/testing with trusted skills."
            )
            UnsafeScriptExecutor._warned = True

    def _get_restricted_env(self, working_dir: Path) -> dict:
        """Get a restricted environment for script execution."""
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
            "PWD": str(working_dir),
        }

        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            env["PATH"] = os.environ["VIRTUAL_ENV"] + "/bin:" + env["PATH"]

        env["PYTHONPATH"] = str(working_dir)

        return env

    def run(
        self,
        script_path: Path,
        arguments: list[str],
        working_dir: Path,
        timeout: int | None = None,
        permissions: SkillPermissions | None = None,
    ) -> ExecutionResult:
        """Execute a script using subprocess (no sandboxing)."""
        self._warn_once()
        timeout = timeout or (permissions.timeout if permissions else None) or self.timeout

        self._validate_script_path(script_path, working_dir)

        if not script_path.exists():
            raise ExecutionError(script_path, "Script file does not exist")
        if not script_path.is_file():
            raise ExecutionError(script_path, "Script path is not a file")

        interpreter_path = shutil.which("python3") or shutil.which("python") or "python3"
        cmd = [interpreter_path, str(script_path)] + arguments
        env = self._get_restricted_env(working_dir)

        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return ExecutionResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Script timed out after {timeout} seconds",
                timed_out=True,
            )
        except FileNotFoundError as e:
            raise ExecutionError(script_path, f"Interpreter not found: {e}")
        except PermissionError as e:
            raise ExecutionError(script_path, f"Permission denied: {e}")
        except Exception as e:
            raise ExecutionError(script_path, str(e))


class WASMScriptExecutor(BaseScriptExecutor):
    """Script executor using DSPy's WASM-based PythonInterpreter.

    Provides true isolation via Deno/Pyodide with fine-grained permission controls.
    Only supports Python scripts.

    Limitations:
        - No access to host filesystem (scripts run in isolated WASM environment)
        - No access to system files like /proc, /sys
        - Scripts that rely on system commands (subprocess) won't work
        - Only pure Python scripts with Pyodide-compatible dependencies work

    Prerequisites:
        Deno must be installed: https://docs.deno.com/runtime/getting_started/installation/
    """

    def __init__(
        self,
        timeout: int = 30,
        allow_network: bool = False,
    ):
        super().__init__(timeout, allow_network)
        self._deno_available = self._check_deno()

    def _check_deno(self) -> bool:
        """Check if Deno is available."""
        return shutil.which("deno") is not None

    def _get_interpreter(
        self,
        script_path: Path,
        permissions: SkillPermissions | None = None,
    ):
        """Get a PythonInterpreter with appropriate permissions.

        Note: The script content is injected directly, so we don't need to mount
        the script file itself. Only explicitly specified file paths (not directories)
        are mounted for reading.
        """
        from dspy.primitives.python_interpreter import PythonInterpreter

        if not self._deno_available:
            raise SecurityError(
                "WASM sandbox mode requires Deno to be installed. "
                "Install it from: https://docs.deno.com/runtime/getting_started/installation/ "
                "or use sandbox_mode='unsafe' for development."
            )

        # Build permission lists - only include files, not directories
        # The script content is injected directly, so no need to mount it
        read_paths = []
        write_paths = []
        env_vars = []
        network_access = []

        if permissions:
            # Only include paths that are files, not directories
            for path in permissions.read_paths:
                p = Path(path)
                if p.exists() and p.is_file():
                    read_paths.append(str(p))
            for path in permissions.write_paths:
                p = Path(path)
                if p.exists() and p.is_file():
                    write_paths.append(str(p))
            env_vars.extend(permissions.env_vars)
            if permissions.network and self.allow_network:
                network_access = ["*"]

        return PythonInterpreter(
            enable_read_paths=read_paths if read_paths else None,
            enable_write_paths=write_paths if write_paths else None,
            enable_env_vars=env_vars if env_vars else None,
            enable_network_access=network_access if network_access else None,
            sync_files=bool(write_paths),
        )

    def run(
        self,
        script_path: Path,
        arguments: list[str],
        working_dir: Path,
        timeout: int | None = None,
        permissions: SkillPermissions | None = None,
    ) -> ExecutionResult:
        """Execute a Python script using WASM sandbox."""
        timeout = timeout or (permissions.timeout if permissions else None) or self.timeout

        self._validate_script_path(script_path, working_dir)

        if script_path.suffix.lower() != ".py":
            raise SecurityError(f"WASM sandbox only supports Python scripts (.py), got: {script_path.suffix}")

        if not script_path.exists():
            raise ExecutionError(script_path, "Script file does not exist")
        if not script_path.is_file():
            raise ExecutionError(script_path, "Script path is not a file")

        try:
            script_content = script_path.read_text()
        except Exception as e:
            raise ExecutionError(script_path, f"Failed to read script: {e}")

        # Prepare the code with sys.argv and capture print output
        script_name = script_path.name
        argv_setup = f"""
import sys
import io
sys.argv = [{script_name!r}, {", ".join(repr(arg) for arg in arguments)}]

# Capture stdout
_original_stdout = sys.stdout
_captured_output = io.StringIO()
sys.stdout = _captured_output
"""
        output_capture = """
# Get captured output
sys.stdout = _original_stdout
_output_text = _captured_output.getvalue()
print(_output_text, end='')
"""
        full_code = argv_setup + script_content + output_capture

        interpreter = self._get_interpreter(script_path, permissions)

        try:
            with interpreter:
                output = interpreter.execute(full_code)

            stdout = str(output) if output is not None else ""

            return ExecutionResult(
                returncode=0,
                stdout=stdout,
                stderr="",
            )

        except SyntaxError as e:
            return ExecutionResult(
                returncode=1,
                stdout="",
                stderr=f"SyntaxError: {e}",
            )
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return ExecutionResult(
                    returncode=-1,
                    stdout="",
                    stderr=f"Script timed out after {timeout} seconds",
                    timed_out=True,
                )
            return ExecutionResult(
                returncode=1,
                stdout="",
                stderr=f"Error: {error_msg}",
            )


def create_script_executor(
    sandbox_mode: SandboxMode = "unsafe",
    timeout: int = 30,
    allow_network: bool = False,
) -> BaseScriptExecutor:
    """Create a script executor based on the sandbox mode.

    Args:
        sandbox_mode: "wasm" for WASM sandbox, "unsafe" for direct subprocess
        timeout: Default timeout in seconds
        allow_network: Whether to allow network access

    Returns:
        A script executor instance
    """
    if sandbox_mode == "wasm":
        return WASMScriptExecutor(timeout=timeout, allow_network=allow_network)
    else:
        return UnsafeScriptExecutor(timeout=timeout, allow_network=allow_network)


# =============================================================================
# SkillsReAct Module
# =============================================================================


class SkillsReAct(Module):
    """A ReAct agent with skill-based capabilities.

    SkillsReAct extends the standard ReAct paradigm by providing access to
    a library of skills that can be discovered, activated, and used during
    the agent's reasoning process. Skills follow the Agent Skills specification
    (agentskills.io) and are defined in SKILL.md files.

    The agent has access to skill meta-tools:
    - list_skills(): Discover available skills with descriptions
    - activate_skill(skill_name): Load full instructions for a skill
    - read_skill_resource(skill_name, resource_type, filename): Read reference/asset files
    - run_skill_script(skill_name, script_name, arguments): Run a Python script (if enabled)

    Args:
        signature: The signature of the module, defining input and output fields.
        skill_dirs: List of directories to scan for skills (subdirectories with SKILL.md).
        tools: Optional additional tools to provide to the agent.
        max_iters: Maximum number of ReAct iterations. Defaults to 20.
        enable_scripts: Whether to enable script execution. Defaults to False (opt-in).
        script_timeout: Timeout in seconds for script execution. Defaults to 30.
        enable_network: Whether scripts can access the network. Defaults to False.
        sandbox_mode: Sandboxing backend - "wasm" (secure, requires Deno) or "unsafe"
            (no sandboxing, for local development only). Defaults to "unsafe".

    Example:
        ```python
        # Basic usage without scripts
        skills_agent = dspy.SkillsReAct(
            signature="request: str -> response: str",
            skill_dirs=["./skills"],
            max_iters=10,
        )

        # With script execution enabled (WASM sandbox)
        skills_agent = dspy.SkillsReAct(
            signature="request: str -> response: str",
            skill_dirs=["./skills"],
            enable_scripts=True,
            sandbox_mode="wasm",  # Requires Deno
        )

        result = skills_agent(request="What process is taking up a lot of CPU usage?")
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        skill_dirs: list[str | Path],
        tools: list[Callable] | None = None,
        max_iters: int = 20,
        enable_scripts: bool = False,
        script_timeout: int = 30,
        enable_network: bool = False,
        sandbox_mode: SandboxMode = "unsafe",
    ):
        super().__init__()

        self._enable_scripts = enable_scripts
        self._script_timeout = script_timeout
        self._enable_network = enable_network
        self._sandbox_mode = sandbox_mode

        # Initialize skill manager and discover skills
        self._skill_manager = SkillManager([Path(d) for d in skill_dirs])
        self._skill_manager.discover()

        # Initialize script executor if scripts are enabled
        self._script_executor: BaseScriptExecutor | None = None
        if enable_scripts:
            self._script_executor = create_script_executor(
                sandbox_mode=sandbox_mode,
                timeout=script_timeout,
                allow_network=enable_network,
            )

        # Create skill meta-tools
        skill_tools = self._create_skill_tools()

        # Combine user tools with skill tools
        all_tools = list(tools or []) + skill_tools

        # Add bash tool if any skill declares allowed-tools with Bash
        bash_tool = self._create_bash_tool()
        if bash_tool:
            all_tools.append(bash_tool)

        # Create the internal ReAct module
        self.react = ReAct(signature=signature, tools=all_tools, max_iters=max_iters)
        self.signature = self.react.signature

    def _any_skill_needs_bash(self) -> bool:
        """Check if any skill declares Bash in allowed-tools."""
        return any(
            skill.allowed_tools and "Bash(" in skill.allowed_tools for skill in self._skill_manager.list_skills()
        )

    def _create_bash_tool(self) -> Tool | None:
        """Create a bash tool scoped to the active skill's allowed-tools.

        Skills can declare allowed-tools in their frontmatter to indicate they need
        shell access. Example: `allowed-tools: Bash(gog:*)`

        Only commands matching the ACTIVE skill's patterns will be allowed to run.
        This only works in 'unsafe' sandbox mode.

        Returns:
            A Tool for bash execution, or None if no skill needs it
        """
        if not self._any_skill_needs_bash():
            return None

        if self._sandbox_mode == "wasm":
            logger.warning(
                "Skills declare Bash tools but sandbox_mode='wasm' cannot run external commands. "
                "Use sandbox_mode='unsafe' for CLI-based skills."
            )
            return None

        # Pattern to extract command prefixes from Bash(command:*)
        bash_pattern = re.compile(r"Bash\(([^:]+):\*\)")

        manager = self._skill_manager
        timeout = self._script_timeout

        def bash(command: str) -> str:
            """Execute a bash command if allowed by the active skill.

            Args:
                command: The shell command to execute

            Returns:
                Command output (stdout + stderr), or an error message.
            """
            # Check active skill
            active_skill = None
            for skill in manager.list_skills():
                if skill.state == SkillState.ACTIVATED:
                    active_skill = skill
                    break

            if not active_skill:
                return "Error: No skill is active. Activate a skill first with activate_skill()."

            if not active_skill.allowed_tools:
                return f"Error: Skill '{active_skill.name}' does not declare any allowed-tools."

            # Parse allowed commands from active skill
            allowed_commands = set(bash_pattern.findall(active_skill.allowed_tools))
            if not allowed_commands:
                return f"Error: Skill '{active_skill.name}' does not allow any bash commands."

            # Get the first word (the command being run)
            cmd_parts = command.strip().split()
            if not cmd_parts:
                return "Error: Empty command"

            base_cmd = cmd_parts[0]

            # Check if command is allowed by active skill
            if base_cmd not in allowed_commands:
                return (
                    f"Error: Command '{base_cmd}' is not allowed by skill '{active_skill.name}'. "
                    f"Allowed commands: {', '.join(sorted(allowed_commands))}"
                )

            # Execute the command
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=active_skill.path,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n[Exit code: {result.returncode}]"
                return output.strip() if output.strip() else "(no output)"
            except subprocess.TimeoutExpired:
                return f"Error: Command timed out after {timeout} seconds"
            except Exception as e:
                return f"Error: {e}"

        return Tool(
            func=bash,
            name="bash",
            desc=(
                "Execute a bash command. Commands are restricted to those allowed "
                "by the currently active skill's allowed-tools declaration. "
                "The skill must be activated first."
            ),
            args={"command": {"type": "string", "description": "The shell command to execute"}},
        )

    def _create_skill_tools(self) -> list[Tool]:
        """Create the skill meta-tools for the agent."""
        manager = self._skill_manager
        enable_scripts = self._enable_scripts
        script_executor = self._script_executor

        def list_skills() -> str:
            """List all available skills with their names and descriptions.

            Returns a formatted list of skills that can be activated for more details.
            Use this to discover what capabilities are available.
            """
            skills = manager.list_skills()
            if not skills:
                return "No skills available."

            lines = ["Available skills:"]
            for skill in skills:
                status = "[ACTIVATED]" if skill.state == SkillState.ACTIVATED else ""
                lines.append(f"- {skill.name}: {skill.description} {status}".strip())

                # Show available resources
                resources = []
                if enable_scripts and skill.has_scripts():
                    scripts = manager.list_scripts(skill.name)
                    if scripts:
                        resources.append(f"scripts: {', '.join(scripts[:5])}")
                        if len(scripts) > 5:
                            resources[-1] += f" (+{len(scripts) - 5} more)"
                if skill.has_references():
                    refs = manager.list_references(skill.name)
                    if refs:
                        resources.append(f"references: {', '.join(refs)}")
                if skill.has_assets():
                    assets = manager.list_assets(skill.name)
                    if assets:
                        resources.append(f"assets: {', '.join(assets[:5])}")
                        if len(assets) > 5:
                            resources[-1] += f" (+{len(assets) - 5} more)"

                if resources:
                    lines.append(f"  Resources: {'; '.join(resources)}")

            return "\n".join(lines)

        def activate_skill(skill_name: str) -> str:
            """Activate a skill to get its full instructions and capabilities.

            Args:
                skill_name: The name of the skill to activate (from list_skills)

            Returns:
                The full instructions for using the skill, or an error message.
            """
            try:
                skill = manager.activate(skill_name)
                result = f"# Skill: {skill.name}\n\n"
                if skill.compatibility:
                    result += f"**Compatibility:** {skill.compatibility}\n\n"
                result += skill.instructions or "(No instructions available)"
                return result
            except SkillNotFoundError as e:
                return str(e)
            except Exception as e:
                return f"Error activating skill '{skill_name}': {e}"

        def read_skill_resource(skill_name: str, resource_type: str, filename: str) -> str:
            """Read a reference or asset file from a skill.

            Args:
                skill_name: The name of the skill
                resource_type: Either 'references' or 'assets'
                filename: The name of the file to read

            Returns:
                The contents of the file, or an error message.
            """
            try:
                path = manager.get_resource_path(skill_name, resource_type, filename)
                # Only read text files
                try:
                    return path.read_text()
                except UnicodeDecodeError:
                    return f"Cannot read binary file: {filename}. Path: {path}"
            except (SkillNotFoundError, ResourceNotFoundError) as e:
                return str(e)
            except ValueError as e:
                return str(e)
            except Exception as e:
                return f"Error reading resource: {e}"

        def run_skill_script(skill_name: str, script_name: str, arguments: str = "") -> str:
            """Run a Python script from an activated skill in a sandboxed environment.

            Scripts perform specific operations like extracting data, validating files,
            or processing documents. Always run a script with --help first to understand
            its usage before running with actual arguments.

            Args:
                skill_name: The name of the skill containing the script
                script_name: The name of the script file (e.g., "cpu_info.py")
                arguments: Space-separated arguments to pass to the script (default: "")

            Returns:
                The script's output (stdout), or an error message if execution fails.
            """
            if not enable_scripts:
                return "Error: Script execution is disabled. Enable it with enable_scripts=True."

            if script_executor is None:
                return "Error: Script executor not initialized."

            # Check if skill exists
            skill = manager.get_skill(skill_name)
            if skill is None:
                available = [s.name for s in manager.list_skills()]
                return (
                    f"Error: Skill '{skill_name}' not found. "
                    f"Available skills: {', '.join(available) if available else 'none'}"
                )

            # Verify skill is activated
            if skill.state != SkillState.ACTIVATED:
                return (
                    f"Error: Skill '{skill_name}' must be activated before running scripts. "
                    f"Use activate_skill('{skill_name}') first."
                )

            # Get script path
            try:
                script_path = manager.get_script_path(skill_name, script_name)
            except ResourceNotFoundError:
                available = manager.list_scripts(skill_name)
                return (
                    f"Error: Script '{script_name}' not found in skill '{skill_name}'. "
                    f"Available scripts: {', '.join(available) if available else 'none'}"
                )
            except SecurityError as e:
                return f"Security error: {e}"
            except SkillNotFoundError as e:
                return f"Error: {e}"

            # Parse arguments
            args = arguments.split() if arguments.strip() else []

            # Execute the script
            try:
                result = script_executor.run(
                    script_path=script_path,
                    arguments=args,
                    working_dir=skill.path,
                    permissions=skill.permissions,
                )

                if result.timed_out:
                    return f"Error: Script timed out after {script_executor.timeout} seconds"

                if result.returncode == 0:
                    output = result.stdout.strip() if result.stdout else "(no output)"
                    return f"Script executed successfully:\n\n{output}"
                else:
                    error_output = result.stderr.strip() if result.stderr else result.stdout.strip()
                    return (
                        f"Script exited with code {result.returncode}:\n\n"
                        f"{error_output if error_output else '(no error output)'}"
                    )

            except SecurityError as e:
                return f"Security error: {e}"
            except ExecutionError as e:
                return f"Execution error: {e}"
            except Exception as e:
                return f"Error running script: {e!s}"

        # Build list of tools
        tools_list = [
            Tool(
                func=list_skills,
                name="list_skills",
                desc="List all available skills with their names and descriptions. Use this to discover what capabilities are available.",
                args={},
            ),
            Tool(
                func=activate_skill,
                name="activate_skill",
                desc="Activate a skill to get its full instructions and capabilities. You must call list_skills first to see available skills.",
                args={"skill_name": {"type": "string", "description": "The name of the skill to activate"}},
            ),
            Tool(
                func=read_skill_resource,
                name="read_skill_resource",
                desc="Read a reference or asset file from an activated skill. Use this to access additional documentation or resources.",
                args={
                    "skill_name": {"type": "string", "description": "The name of the skill"},
                    "resource_type": {"type": "string", "description": "Either 'references' or 'assets'"},
                    "filename": {"type": "string", "description": "The name of the file to read"},
                },
            ),
        ]

        # Add run_skill_script tool if scripts are enabled
        if enable_scripts:
            tools_list.append(
                Tool(
                    func=run_skill_script,
                    name="run_skill_script",
                    desc=(
                        "Run a Python script from an activated skill. Scripts perform specific operations "
                        "like extracting data or processing files. Always run with --help first to see usage. "
                        "The skill must be activated before running its scripts."
                    ),
                    args={
                        "skill_name": {"type": "string", "description": "The name of the skill containing the script"},
                        "script_name": {"type": "string", "description": "The script filename (e.g., 'cpu_info.py')"},
                        "arguments": {
                            "type": "string",
                            "description": "Space-separated arguments (e.g., '--help' or '--json')",
                        },
                    },
                )
            )

        return tools_list

    @property
    def skill_manager(self) -> SkillManager:
        """Access the skill manager for direct skill operations."""
        return self._skill_manager

    def forward(self, **input_args):
        """Execute the SkillsReAct agent.

        The agent will use the ReAct loop to reason about the task,
        potentially discovering and activating skills as needed.

        Args:
            **input_args: Input arguments matching the signature's input fields.

        Returns:
            A dspy.Prediction with the output fields and trajectory.
        """
        return self.react(**input_args)

    async def aforward(self, **input_args):
        """Async version of forward().

        Args:
            **input_args: Input arguments matching the signature's input fields.

        Returns:
            A dspy.Prediction with the output fields and trajectory.
        """
        return await self.react.acall(**input_args)
