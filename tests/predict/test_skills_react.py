"""Tests for the SkillsReAct module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module components directly for unit testing
from dspy.predict.skills_react import (
    ExecutionError,
    ExecutionResult,
    ParseError,
    ResourceNotFoundError,
    SecurityError,
    Skill,
    SkillManager,
    SkillNotFoundError,
    SkillPermissions,
    SkillsReAct,
    SkillState,
    UnsafeScriptExecutor,
    ValidationError,
    WASMScriptExecutor,
    create_script_executor,
    parse_frontmatter,
    read_skill,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_skill_content():
    """Sample SKILL.md content for testing."""
    return """---
name: test-skill
description: A test skill for unit testing
license: MIT
compatibility: Python 3.9+
metadata:
  author: test
  version: "1.0"
---

# Test Skill

This is the instruction body for the test skill.

## Usage

Use this skill when testing.
"""


@pytest.fixture
def minimal_skill_content():
    """Minimal valid SKILL.md content."""
    return """---
name: minimal
description: Minimal test skill
---

Instructions here.
"""


@pytest.fixture
def skill_with_scripts_content():
    """SKILL.md content for a skill with scripts."""
    return """---
name: scripted-skill
description: A skill with Python scripts
permissions:
  read_paths: ["/tmp"]
  write_paths: []
  env_vars: ["HOME", "PATH"]
  network: false
  timeout: 60
---

# Scripted Skill

This skill has executable Python scripts.
"""


@pytest.fixture
def skill_with_bash_content():
    """SKILL.md content for a skill requiring bash access."""
    return """---
name: cli-skill
description: A skill that uses CLI tools
allowed-tools: Bash(mytool:*)
---

# CLI Skill

Use `mytool` command for operations.
"""


@pytest.fixture
def skill_with_multi_bash_content():
    """SKILL.md content for a skill requiring multiple bash commands."""
    return """---
name: multi-cli-skill
description: A skill that uses multiple CLI tools
allowed-tools: Bash(echo:*) Bash(cat:*)
---

# Multi CLI Skill

Use `echo` and `cat` commands.
"""


@pytest.fixture
def temp_skills_dir(
    sample_skill_content,
    minimal_skill_content,
    skill_with_scripts_content,
    skill_with_bash_content,
):
    """Create a temporary directory with sample skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)

        # Create test-skill
        skill1_dir = skills_dir / "test-skill"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text(sample_skill_content)

        # Create references directory with a file
        refs_dir = skill1_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "guide.md").write_text("# Reference Guide\n\nSome reference content.")

        # Create assets directory with files
        assets_dir = skill1_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "config.json").write_text('{"key": "value"}')
        subdir = assets_dir / "templates"
        subdir.mkdir()
        (subdir / "template.txt").write_text("Template content")

        # Create minimal skill
        skill2_dir = skills_dir / "minimal-skill"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text(minimal_skill_content)

        # Create skill with scripts
        skill3_dir = skills_dir / "scripted-skill"
        skill3_dir.mkdir()
        (skill3_dir / "SKILL.md").write_text(skill_with_scripts_content)

        # Create scripts directory with Python scripts
        scripts_dir = skill3_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "hello.py").write_text('print("Hello from script!")')
        (scripts_dir / "echo_args.py").write_text('import sys\nprint("Args:", sys.argv[1:])')
        (scripts_dir / "failing_script.py").write_text('import sys\nprint("Error!", file=sys.stderr)\nsys.exit(1)')
        (scripts_dir / "syntax_error.py").write_text("def broken(\n")  # Invalid Python

        # Create skill with bash allowed-tools
        skill4_dir = skills_dir / "cli-skill"
        skill4_dir.mkdir()
        (skill4_dir / "SKILL.md").write_text(skill_with_bash_content)

        yield skills_dir


# =============================================================================
# Parser Tests
# =============================================================================


class TestParseFrontmatter:
    """Tests for parse_frontmatter function."""

    def test_valid_frontmatter(self, sample_skill_content):
        """Test parsing valid frontmatter."""
        metadata, body = parse_frontmatter(sample_skill_content)

        assert metadata["name"] == "test-skill"
        assert metadata["description"] == "A test skill for unit testing"
        assert metadata["license"] == "MIT"
        assert "# Test Skill" in body

    def test_missing_frontmatter(self):
        """Test error when frontmatter is missing."""
        with pytest.raises(ParseError, match="must start with YAML frontmatter"):
            parse_frontmatter("# No frontmatter here")

    def test_unclosed_frontmatter(self):
        """Test error when frontmatter is not closed."""
        with pytest.raises(ParseError, match="not properly closed"):
            parse_frontmatter("---\nname: test\nNo closing delimiter")

    def test_invalid_yaml(self):
        """Test error with invalid YAML."""
        with pytest.raises(ParseError, match="Invalid YAML"):
            parse_frontmatter("---\n: invalid yaml :\n---\nbody")


class TestReadSkill:
    """Tests for read_skill function."""

    def test_read_skill_metadata_only(self, temp_skills_dir):
        """Test reading skill with metadata only."""
        skill_dir = temp_skills_dir / "test-skill"
        skill = read_skill(skill_dir, load_instructions=False)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert skill.state == SkillState.DISCOVERED
        assert skill.instructions is None
        assert skill.license == "MIT"

    def test_read_skill_with_instructions(self, temp_skills_dir):
        """Test reading skill with full instructions."""
        skill_dir = temp_skills_dir / "test-skill"
        skill = read_skill(skill_dir, load_instructions=True)

        assert skill.name == "test-skill"
        assert skill.state == SkillState.ACTIVATED
        assert skill.instructions is not None
        assert "# Test Skill" in skill.instructions

    def test_read_skill_missing_name(self, temp_skills_dir):
        """Test error when name field is missing."""
        skill_dir = temp_skills_dir / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: test\n---\nbody")

        with pytest.raises(ValidationError, match="Missing required field.*name"):
            read_skill(skill_dir)

    def test_read_skill_missing_description(self, temp_skills_dir):
        """Test error when description field is missing."""
        skill_dir = temp_skills_dir / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\nbody")

        with pytest.raises(ValidationError, match="Missing required field.*description"):
            read_skill(skill_dir)

    def test_read_skill_no_skill_md(self, temp_skills_dir):
        """Test error when SKILL.md is missing."""
        skill_dir = temp_skills_dir / "empty-skill"
        skill_dir.mkdir()

        with pytest.raises(ParseError, match="SKILL.md not found"):
            read_skill(skill_dir)


# =============================================================================
# Skill Data Model Tests
# =============================================================================


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_skill_properties(self, temp_skills_dir):
        """Test Skill directory properties."""
        skill_dir = temp_skills_dir / "test-skill"
        skill = read_skill(skill_dir)

        assert isinstance(skill, Skill)
        assert skill.has_references()
        assert skill.has_assets()
        assert skill.references_dir == skill_dir / "references"
        assert skill.assets_dir == skill_dir / "assets"

    def test_skill_without_resources(self, temp_skills_dir):
        """Test Skill without resource directories."""
        skill_dir = temp_skills_dir / "minimal-skill"
        skill = read_skill(skill_dir)

        assert isinstance(skill, Skill)
        assert not skill.has_references()
        assert not skill.has_assets()
        assert skill.references_dir is None
        assert skill.assets_dir is None


# =============================================================================
# SkillManager Tests
# =============================================================================


class TestSkillManager:
    """Tests for the SkillManager class."""

    def test_discover_skills(self, temp_skills_dir):
        """Test skill discovery."""
        manager = SkillManager([temp_skills_dir])
        discovered = manager.discover()

        assert len(discovered) == 4
        assert "test-skill" in discovered
        assert "minimal" in discovered
        assert "scripted-skill" in discovered
        assert "cli-skill" in discovered

    def test_list_skills(self, temp_skills_dir):
        """Test listing all skills."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()
        skills = manager.list_skills()

        assert len(skills) == 4
        names = [s.name for s in skills]
        assert "test-skill" in names
        assert "minimal" in names
        assert "scripted-skill" in names
        assert "cli-skill" in names

    def test_get_skill(self, temp_skills_dir):
        """Test getting a specific skill."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        skill = manager.get_skill("test-skill")
        assert skill is not None
        assert skill.name == "test-skill"

        assert manager.get_skill("nonexistent") is None

    def test_activate_skill(self, temp_skills_dir):
        """Test skill activation."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        skill = manager.activate("test-skill")
        assert skill.state == SkillState.ACTIVATED
        assert skill.instructions is not None
        assert "# Test Skill" in skill.instructions

    def test_activate_nonexistent_skill(self, temp_skills_dir):
        """Test error when activating nonexistent skill."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(SkillNotFoundError) as exc_info:
            manager.activate("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "test-skill" in str(exc_info.value)  # Shows available

    def test_list_references(self, temp_skills_dir):
        """Test listing skill references."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        refs = manager.list_references("test-skill")
        assert "guide.md" in refs

        # Skill without references
        refs = manager.list_references("minimal")
        assert refs == []

    def test_list_assets(self, temp_skills_dir):
        """Test listing skill assets."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        assets = manager.list_assets("test-skill")
        assert "config.json" in assets
        assert "templates/template.txt" in assets

    def test_get_resource_path(self, temp_skills_dir):
        """Test getting resource paths."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        # Reference file
        path = manager.get_resource_path("test-skill", "references", "guide.md")
        assert path.exists()
        assert path.name == "guide.md"

        # Asset file
        path = manager.get_resource_path("test-skill", "assets", "config.json")
        assert path.exists()

        # Nested asset
        path = manager.get_resource_path("test-skill", "assets", "templates/template.txt")
        assert path.exists()

    def test_get_resource_path_not_found(self, temp_skills_dir):
        """Test error when resource doesn't exist."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(ResourceNotFoundError):
            manager.get_resource_path("test-skill", "references", "nonexistent.md")

    def test_get_resource_path_invalid_type(self, temp_skills_dir):
        """Test error with invalid resource type."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(ValueError, match="Invalid resource_type"):
            manager.get_resource_path("test-skill", "scripts", "test.py")

    def test_discover_nonexistent_directory(self):
        """Test discovery with nonexistent directory."""
        manager = SkillManager([Path("/nonexistent/path")])
        discovered = manager.discover()
        assert discovered == []

    def test_discover_multiple_directories(self, temp_skills_dir):
        """Test discovery from multiple directories."""
        with tempfile.TemporaryDirectory() as tmpdir2:
            # Create another skill in second directory
            skill_dir = Path(tmpdir2) / "other-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: other\ndescription: Other skill\n---\nBody")

            manager = SkillManager([temp_skills_dir, Path(tmpdir2)])
            discovered = manager.discover()

            assert len(discovered) == 5  # 4 from temp_skills_dir + 1 from tmpdir2
            assert "other" in discovered


# =============================================================================
# SkillsReAct Integration Tests
# =============================================================================


class TestSkillsReActIntegration:
    """Integration tests for SkillsReAct module."""

    def test_initialization(self, temp_skills_dir):
        """Test SkillsReAct initialization."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            max_iters=5,
        )

        assert agent.skill_manager is not None
        skills = agent.skill_manager.list_skills()
        assert len(skills) == 4

    def test_skill_tools_created(self, temp_skills_dir):
        """Test that skill meta-tools are created."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        # Check that the ReAct module has the skill tools
        tool_names = list(agent.react.tools.keys())
        assert "list_skills" in tool_names
        assert "activate_skill" in tool_names
        assert "read_skill_resource" in tool_names
        assert "finish" in tool_names  # ReAct's built-in finish tool

    def test_list_skills_tool(self, temp_skills_dir):
        """Test the list_skills tool output."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        # Call the list_skills tool directly
        list_skills_tool = agent.react.tools["list_skills"]
        result = list_skills_tool()

        assert "test-skill" in result
        assert "minimal" in result
        assert "A test skill for unit testing" in result

    def test_activate_skill_tool(self, temp_skills_dir):
        """Test the activate_skill tool output."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        activate_tool = agent.react.tools["activate_skill"]
        result = activate_tool(skill_name="test-skill")

        assert "# Skill: test-skill" in result
        assert "# Test Skill" in result
        assert "Use this skill when testing" in result

    def test_activate_nonexistent_skill_tool(self, temp_skills_dir):
        """Test activate_skill tool with nonexistent skill."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        activate_tool = agent.react.tools["activate_skill"]
        result = activate_tool(skill_name="nonexistent")

        assert "not found" in result.lower()

    def test_read_skill_resource_tool(self, temp_skills_dir):
        """Test the read_skill_resource tool."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        read_tool = agent.react.tools["read_skill_resource"]
        result = read_tool(
            skill_name="test-skill",
            resource_type="references",
            filename="guide.md",
        )

        assert "# Reference Guide" in result
        assert "Some reference content" in result

    def test_read_nonexistent_resource(self, temp_skills_dir):
        """Test read_skill_resource with nonexistent file."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
        )

        read_tool = agent.react.tools["read_skill_resource"]
        result = read_tool(
            skill_name="test-skill",
            resource_type="references",
            filename="nonexistent.md",
        )

        assert "not found" in result.lower()

    def test_with_additional_tools(self, temp_skills_dir):
        """Test SkillsReAct with additional user tools."""

        def my_tool(x: str) -> str:
            """A custom tool."""
            return f"Result: {x}"

        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            tools=[my_tool],
        )

        tool_names = list(agent.react.tools.keys())
        assert "my_tool" in tool_names
        assert "list_skills" in tool_names

    def test_empty_skills_directory(self):
        """Test with an empty skills directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SkillsReAct(
                signature="question: str -> answer: str",
                skill_dirs=[tmpdir],
            )

            list_tool = agent.react.tools["list_skills"]
            result = list_tool()

            assert "No skills available" in result


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurity:
    """Security-related tests."""

    def test_path_traversal_prevention(self, temp_skills_dir):
        """Test that path traversal attacks are prevented."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        # Try to escape the resource directory
        with pytest.raises(ResourceNotFoundError):
            manager.get_resource_path("test-skill", "references", "../SKILL.md")

        with pytest.raises(ResourceNotFoundError):
            manager.get_resource_path("test-skill", "assets", "../../SKILL.md")

    def test_script_path_traversal_prevention(self, temp_skills_dir):
        """Test that script path traversal attacks are prevented."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        # Try to escape the scripts directory
        with pytest.raises(SecurityError):
            manager.get_script_path("scripted-skill", "../SKILL.md")

        with pytest.raises(SecurityError):
            manager.get_script_path("scripted-skill", "../../other-skill/SKILL.md")

    def test_only_py_scripts_allowed(self, temp_skills_dir):
        """Test that only .py scripts are allowed."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(SecurityError, match="Only Python scripts"):
            manager.get_script_path("scripted-skill", "malicious.sh")

        with pytest.raises(SecurityError, match="Only Python scripts"):
            manager.get_script_path("scripted-skill", "script.rb")


# =============================================================================
# SkillPermissions Tests
# =============================================================================


class TestSkillPermissions:
    """Tests for SkillPermissions dataclass."""

    def test_default_permissions(self):
        """Test default permission values."""
        perms = SkillPermissions()
        assert perms.read_paths == []
        assert perms.write_paths == []
        assert perms.env_vars == []
        assert perms.network is False
        assert perms.timeout == 30

    def test_custom_permissions(self):
        """Test custom permission values."""
        perms = SkillPermissions(
            read_paths=["/tmp", "/data"],
            write_paths=["/output"],
            env_vars=["HOME", "PATH"],
            network=True,
            timeout=60,
        )
        assert perms.read_paths == ["/tmp", "/data"]
        assert perms.write_paths == ["/output"]
        assert perms.env_vars == ["HOME", "PATH"]
        assert perms.network is True
        assert perms.timeout == 60

    def test_permissions_parsed_from_skill(self, temp_skills_dir):
        """Test that permissions are parsed from SKILL.md frontmatter."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        skill = manager.get_skill("scripted-skill")
        assert skill is not None
        assert skill.permissions is not None
        assert "/tmp" in skill.permissions.read_paths
        assert skill.permissions.write_paths == []
        assert "HOME" in skill.permissions.env_vars
        assert skill.permissions.network is False
        assert skill.permissions.timeout == 60


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(returncode=0, stdout="output", stderr="")
        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.timed_out is False

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(returncode=1, stdout="", stderr="error message")
        assert result.returncode == 1
        assert result.stderr == "error message"
        assert result.timed_out is False

    def test_timeout_result(self):
        """Test timeout execution result."""
        result = ExecutionResult(returncode=-1, stdout="", stderr="timeout", timed_out=True)
        assert result.timed_out is True
        assert result.returncode == -1


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_execution_error(self):
        """Test ExecutionError exception."""
        err = ExecutionError(Path("/test/script.py"), "failed to run", "some stderr")
        assert "script.py" in str(err)
        assert "failed to run" in str(err)
        assert err.script_path == Path("/test/script.py")
        assert err.reason == "failed to run"
        assert err.stderr == "some stderr"

    def test_security_error(self):
        """Test SecurityError exception."""
        err = SecurityError("Access denied")
        assert "Access denied" in str(err)
        assert isinstance(err, Exception)

    def test_skill_not_found_error_with_available(self):
        """Test SkillNotFoundError shows available skills."""
        err = SkillNotFoundError("missing", available=["skill1", "skill2"])
        assert "missing" in str(err)
        assert "skill1" in str(err)
        assert "skill2" in str(err)

    def test_validation_error_with_multiple_errors(self):
        """Test ValidationError with multiple errors."""
        err = ValidationError("Multiple errors", errors=["error1", "error2"])
        assert err.errors == ["error1", "error2"]


# =============================================================================
# Script Manager Tests
# =============================================================================


class TestSkillManagerScripts:
    """Tests for SkillManager script-related methods."""

    def test_list_scripts(self, temp_skills_dir):
        """Test listing scripts from a skill."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        scripts = manager.list_scripts("scripted-skill")
        assert "hello.py" in scripts
        assert "echo_args.py" in scripts
        assert "failing_script.py" in scripts
        assert len(scripts) >= 3

    def test_list_scripts_no_scripts_dir(self, temp_skills_dir):
        """Test listing scripts when skill has no scripts directory."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        scripts = manager.list_scripts("minimal")
        assert scripts == []

    def test_list_scripts_nonexistent_skill(self, temp_skills_dir):
        """Test listing scripts for nonexistent skill."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(SkillNotFoundError):
            manager.list_scripts("nonexistent")

    def test_get_script_path(self, temp_skills_dir):
        """Test getting script path."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        path = manager.get_script_path("scripted-skill", "hello.py")
        assert path.exists()
        assert path.name == "hello.py"
        assert path.suffix == ".py"

    def test_get_script_path_nonexistent(self, temp_skills_dir):
        """Test getting nonexistent script path."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        with pytest.raises(ResourceNotFoundError):
            manager.get_script_path("scripted-skill", "nonexistent.py")

    def test_skill_has_scripts(self, temp_skills_dir):
        """Test has_scripts property."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        scripted = manager.get_skill("scripted-skill")
        minimal = manager.get_skill("minimal")

        assert scripted.has_scripts() is True
        assert scripted.scripts_dir is not None
        assert minimal.has_scripts() is False
        assert minimal.scripts_dir is None


# =============================================================================
# UnsafeScriptExecutor Tests
# =============================================================================


class TestUnsafeScriptExecutor:
    """Tests for UnsafeScriptExecutor."""

    def test_executor_creation(self):
        """Test creating an UnsafeScriptExecutor."""
        executor = UnsafeScriptExecutor(timeout=60, allow_network=True)
        assert executor.timeout == 60
        assert executor.allow_network is True

    def test_run_simple_script(self, temp_skills_dir):
        """Test running a simple Python script."""
        executor = UnsafeScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "hello.py"
        working_dir = temp_skills_dir / "scripted-skill"

        result = executor.run(
            script_path=script_path,
            arguments=[],
            working_dir=working_dir,
        )

        assert result.returncode == 0
        assert "Hello from script!" in result.stdout

    def test_run_script_with_args(self, temp_skills_dir):
        """Test running a script with arguments."""
        executor = UnsafeScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "echo_args.py"
        working_dir = temp_skills_dir / "scripted-skill"

        result = executor.run(
            script_path=script_path,
            arguments=["arg1", "arg2", "arg3"],
            working_dir=working_dir,
        )

        assert result.returncode == 0
        assert "arg1" in result.stdout
        assert "arg2" in result.stdout

    def test_run_failing_script(self, temp_skills_dir):
        """Test running a script that exits with non-zero code."""
        executor = UnsafeScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "failing_script.py"
        working_dir = temp_skills_dir / "scripted-skill"

        result = executor.run(
            script_path=script_path,
            arguments=[],
            working_dir=working_dir,
        )

        assert result.returncode == 1
        assert "Error!" in result.stderr

    def test_run_nonexistent_script(self, temp_skills_dir):
        """Test running a nonexistent script."""
        executor = UnsafeScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "nonexistent.py"
        working_dir = temp_skills_dir / "scripted-skill"

        with pytest.raises(ExecutionError, match="does not exist"):
            executor.run(
                script_path=script_path,
                arguments=[],
                working_dir=working_dir,
            )

    def test_script_timeout(self, temp_skills_dir):
        """Test script execution timeout."""
        # Create a slow script
        slow_script = temp_skills_dir / "scripted-skill" / "scripts" / "slow.py"
        slow_script.write_text("import time; time.sleep(10); print('done')")

        executor = UnsafeScriptExecutor(timeout=1)
        working_dir = temp_skills_dir / "scripted-skill"

        result = executor.run(
            script_path=slow_script,
            arguments=[],
            working_dir=working_dir,
        )

        assert result.timed_out is True
        assert result.returncode == -1
        assert "timed out" in result.stderr.lower()

    def test_path_escape_prevention(self, temp_skills_dir):
        """Test that scripts outside skill directory are rejected."""
        executor = UnsafeScriptExecutor(timeout=30)
        # Try to run a script outside the skill directory
        script_path = temp_skills_dir / "other_script.py"
        script_path.write_text("print('should not run')")
        working_dir = temp_skills_dir / "scripted-skill"

        with pytest.raises(SecurityError):
            executor.run(
                script_path=script_path,
                arguments=[],
                working_dir=working_dir,
            )

    def test_warns_once(self, temp_skills_dir, caplog):
        """Test that unsafe mode warning is logged only once."""
        # Reset the warning flag for this test
        UnsafeScriptExecutor._warned = False

        executor = UnsafeScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "hello.py"
        working_dir = temp_skills_dir / "scripted-skill"

        # First run should log warning
        executor.run(script_path=script_path, arguments=[], working_dir=working_dir)

        # Second run should not log another warning
        executor.run(script_path=script_path, arguments=[], working_dir=working_dir)

        # Check that warning appeared (implementation detail, may need adjustment)
        assert UnsafeScriptExecutor._warned is True


# =============================================================================
# WASMScriptExecutor Tests
# =============================================================================


class TestWASMScriptExecutor:
    """Tests for WASMScriptExecutor."""

    def test_executor_creation(self):
        """Test creating a WASMScriptExecutor."""
        executor = WASMScriptExecutor(timeout=60, allow_network=False)
        assert executor.timeout == 60
        assert executor.allow_network is False

    def test_deno_check(self):
        """Test Deno availability check."""
        executor = WASMScriptExecutor()
        # This will be True or False depending on system
        assert isinstance(executor._deno_available, bool)

    @pytest.mark.skipif(shutil.which("deno") is None, reason="Deno not installed")
    def test_run_simple_script_wasm(self, temp_skills_dir):
        """Test running a simple script in WASM sandbox."""
        executor = WASMScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "hello.py"
        working_dir = temp_skills_dir / "scripted-skill"

        result = executor.run(
            script_path=script_path,
            arguments=[],
            working_dir=working_dir,
        )

        assert result.returncode == 0
        assert "Hello from script!" in result.stdout

    def test_rejects_non_python(self, temp_skills_dir):
        """Test that non-Python scripts are rejected."""
        executor = WASMScriptExecutor(timeout=30)
        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "test.sh"
        script_path.write_text("echo hello")
        working_dir = temp_skills_dir / "scripted-skill"

        with pytest.raises(SecurityError, match="only supports Python"):
            executor.run(
                script_path=script_path,
                arguments=[],
                working_dir=working_dir,
            )

    def test_no_deno_error(self, temp_skills_dir):
        """Test error when Deno is not available."""
        executor = WASMScriptExecutor(timeout=30)
        executor._deno_available = False  # Force unavailable

        script_path = temp_skills_dir / "scripted-skill" / "scripts" / "hello.py"
        working_dir = temp_skills_dir / "scripted-skill"

        with pytest.raises(SecurityError, match="requires Deno"):
            executor.run(
                script_path=script_path,
                arguments=[],
                working_dir=working_dir,
            )


# =============================================================================
# create_script_executor Tests
# =============================================================================


class TestCreateScriptExecutor:
    """Tests for create_script_executor factory function."""

    def test_create_unsafe_executor(self):
        """Test creating unsafe executor."""
        executor = create_script_executor(sandbox_mode="unsafe", timeout=45)
        assert isinstance(executor, UnsafeScriptExecutor)
        assert executor.timeout == 45

    def test_create_wasm_executor(self):
        """Test creating WASM executor."""
        executor = create_script_executor(sandbox_mode="wasm", timeout=60)
        assert isinstance(executor, WASMScriptExecutor)
        assert executor.timeout == 60

    def test_default_is_unsafe(self):
        """Test that default mode is unsafe."""
        executor = create_script_executor()
        assert isinstance(executor, UnsafeScriptExecutor)


# =============================================================================
# Allowed Tools / Bash Tool Tests
# =============================================================================


class TestAllowedToolsParsing:
    """Tests for allowed-tools parsing from SKILL.md."""

    def test_allowed_tools_parsed(self, temp_skills_dir):
        """Test that allowed-tools is parsed from frontmatter."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        skill = manager.get_skill("cli-skill")
        assert skill is not None
        assert skill.allowed_tools == "Bash(mytool:*)"

    def test_skill_without_allowed_tools(self, temp_skills_dir):
        """Test skill without allowed-tools."""
        manager = SkillManager([temp_skills_dir])
        manager.discover()

        skill = manager.get_skill("test-skill")
        assert skill.allowed_tools is None


class TestBashTool:
    """Tests for bash tool creation and execution."""

    def test_any_skill_needs_bash(self, temp_skills_dir):
        """Test _any_skill_needs_bash detection."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        # Should detect cli-skill has Bash in allowed-tools
        assert agent._any_skill_needs_bash() is True

    def test_bash_tool_created_when_needed(self, temp_skills_dir):
        """Test that bash tool is created when skills need it."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        tool_names = list(agent.react.tools.keys())
        assert "bash" in tool_names

    def test_bash_tool_not_created_when_no_skills_need_it(self):
        """Test that bash tool is not created when no skills need it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill without allowed-tools
            skill_dir = Path(tmpdir) / "simple-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: simple\ndescription: Simple skill\n---\nBody")

            agent = SkillsReAct(
                signature="question: str -> answer: str",
                skill_dirs=[tmpdir],
                sandbox_mode="unsafe",
            )

            tool_names = list(agent.react.tools.keys())
            assert "bash" not in tool_names

    def test_bash_tool_requires_active_skill(self, temp_skills_dir):
        """Test that bash tool requires an activated skill."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        bash_tool = agent.react.tools.get("bash")
        assert bash_tool is not None

        # Try to run without activating a skill
        result = bash_tool(command="mytool --help")
        assert "No skill is active" in result

    def test_bash_tool_checks_allowed_commands(self, temp_skills_dir):
        """Test that bash tool only allows commands from active skill."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        # Activate the cli-skill (allows mytool)
        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="cli-skill")

        bash_tool = agent.react.tools["bash"]

        # Try to run a non-allowed command
        result = bash_tool(command="rm -rf /")
        assert "not allowed" in result.lower()
        assert "mytool" in result  # Should show allowed commands

    def test_bash_tool_allows_declared_commands(self, temp_skills_dir):
        """Test that bash tool allows declared commands."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        # Activate the cli-skill
        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="cli-skill")

        bash_tool = agent.react.tools["bash"]

        # Mock the command execution since mytool doesn't exist
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="mytool output",
                stderr="",
                returncode=0,
            )

            result = bash_tool(command="mytool --version")

            # Command should be executed
            mock_run.assert_called_once()
            assert "mytool output" in result

    def test_bash_tool_not_created_in_wasm_mode(self, temp_skills_dir, caplog):
        """Test that bash tool warns and is not created in WASM mode."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="wasm",
        )

        tool_names = list(agent.react.tools.keys())
        # Bash tool should not be available in WASM mode
        assert "bash" not in tool_names

    def test_bash_tool_skill_without_allowed_tools(self, temp_skills_dir):
        """Test bash tool with skill that has no allowed-tools."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            sandbox_mode="unsafe",
        )

        # Activate test-skill (no allowed-tools)
        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="test-skill")

        bash_tool = agent.react.tools["bash"]
        result = bash_tool(command="echo hello")

        assert "does not declare any allowed-tools" in result


# =============================================================================
# Script Execution Tool Tests
# =============================================================================


class TestRunSkillScriptTool:
    """Tests for the run_skill_script tool."""

    def test_scripts_disabled_by_default(self, temp_skills_dir):
        """Test that scripts are disabled by default."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            # enable_scripts=False is default
        )

        # run_skill_script should not be in tools
        tool_names = list(agent.react.tools.keys())
        assert "run_skill_script" not in tool_names

    def test_scripts_enabled(self, temp_skills_dir):
        """Test that run_skill_script tool is available when enabled."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        tool_names = list(agent.react.tools.keys())
        assert "run_skill_script" in tool_names

    def test_run_script_requires_activation(self, temp_skills_dir):
        """Test that scripts require skill activation first."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="scripted-skill",
            script_name="hello.py",
            arguments="",
        )

        assert "must be activated" in result

    def test_run_script_success(self, temp_skills_dir):
        """Test successful script execution."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        # Activate the skill first
        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="scripted-skill")

        # Run the script
        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="scripted-skill",
            script_name="hello.py",
            arguments="",
        )

        assert "successfully" in result.lower()
        assert "Hello from script!" in result

    def test_run_script_with_arguments(self, temp_skills_dir):
        """Test script execution with arguments."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="scripted-skill")

        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="scripted-skill",
            script_name="echo_args.py",
            arguments="--flag value1 value2",
        )

        assert "--flag" in result
        assert "value1" in result

    def test_run_script_failure(self, temp_skills_dir):
        """Test script execution failure."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="scripted-skill")

        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="scripted-skill",
            script_name="failing_script.py",
            arguments="",
        )

        assert "exited with code 1" in result

    def test_run_nonexistent_script(self, temp_skills_dir):
        """Test running a nonexistent script."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        activate_tool = agent.react.tools["activate_skill"]
        activate_tool(skill_name="scripted-skill")

        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="scripted-skill",
            script_name="nonexistent.py",
            arguments="",
        )

        assert "not found" in result.lower()

    def test_run_script_nonexistent_skill(self, temp_skills_dir):
        """Test running script from nonexistent skill."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        run_tool = agent.react.tools["run_skill_script"]
        result = run_tool(
            skill_name="nonexistent-skill",
            script_name="hello.py",
            arguments="",
        )

        assert "not found" in result.lower()


# =============================================================================
# SkillsReAct Configuration Tests
# =============================================================================


class TestSkillsReActConfiguration:
    """Tests for SkillsReAct configuration options."""

    def test_custom_timeout(self, temp_skills_dir):
        """Test custom script timeout."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            script_timeout=120,
            sandbox_mode="unsafe",
        )

        assert agent._script_timeout == 120
        assert agent._script_executor.timeout == 120

    def test_network_enabled(self, temp_skills_dir):
        """Test network access configuration."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            enable_network=True,
            sandbox_mode="unsafe",
        )

        assert agent._enable_network is True
        assert agent._script_executor.allow_network is True

    def test_sandbox_mode_wasm(self, temp_skills_dir):
        """Test WASM sandbox mode configuration."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="wasm",
        )

        assert agent._sandbox_mode == "wasm"
        assert isinstance(agent._script_executor, WASMScriptExecutor)

    def test_sandbox_mode_unsafe(self, temp_skills_dir):
        """Test unsafe sandbox mode configuration."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=True,
            sandbox_mode="unsafe",
        )

        assert agent._sandbox_mode == "unsafe"
        assert isinstance(agent._script_executor, UnsafeScriptExecutor)

    def test_no_executor_when_scripts_disabled(self, temp_skills_dir):
        """Test that no executor is created when scripts are disabled."""
        agent = SkillsReAct(
            signature="question: str -> answer: str",
            skill_dirs=[temp_skills_dir],
            enable_scripts=False,
        )

        assert agent._script_executor is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
