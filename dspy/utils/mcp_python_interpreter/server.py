"""
MCP Python Interpreter

A Model Context Protocol server for interacting with Python environments
and executing Python code. All operations are confined to a specified working directory
or allowed system-wide if explicitly enabled.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Import FastMCP for building our server
from mcp.server.fastmcp import FastMCP

# Parse command line arguments to get the working directory
# Use a default value that works when run via uvx
parser = argparse.ArgumentParser(description="MCP Python Interpreter")
parser.add_argument(
    "--dir",
    type=str,
    default=os.getcwd(),
    help="Working directory for code execution and file operations",
)
parser.add_argument(
    "--python-path",
    type=str,
    default=None,
    help="Custom Python interpreter path to use as default",
)
args, unknown = parser.parse_known_args()

# Check if system-wide access is enabled via environment variable
ALLOW_SYSTEM_ACCESS = os.environ.get("MCP_ALLOW_SYSTEM_ACCESS", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Set and create working directory
WORKING_DIR = Path(args.dir).absolute()
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Set default Python path
DEFAULT_PYTHON_PATH = args.python_path if args.python_path else sys.executable

# Print startup message to stderr (doesn't interfere with MCP protocol)
print(f"MCP Python Interpreter starting in directory: {WORKING_DIR}", file=sys.stderr)
print(f"Using default Python interpreter: {DEFAULT_PYTHON_PATH}", file=sys.stderr)
print(
    f"System-wide file access: {'ENABLED' if ALLOW_SYSTEM_ACCESS else 'DISABLED'}",
    file=sys.stderr,
)

# Create our MCP server
mcp = FastMCP(
    "Python Interpreter",
    description="Execute Python code in sandbox",
    dependencies=["mcp[cli]"],
)

# Persistent namespace for user-defined functions
GLOBAL_FUNCTIONS_NS = {}

# Per-session file for registered functions
SESSION_FUNCTIONS_FILE = WORKING_DIR / ".session_functions.py"

# Clean up session file on shutdown
import atexit


def _cleanup_session_file():
    print("Cleaning up session file")
    try:
        if SESSION_FUNCTIONS_FILE.exists():
            SESSION_FUNCTIONS_FILE.unlink()
    except Exception:
        pass


atexit.register(_cleanup_session_file)

# ============================================================================
# Helper functions
# ============================================================================


def is_path_allowed(path: Path) -> bool:
    """
    Check if a path is allowed based on security settings.

    Args:
        path: Path to check

    Returns:
        bool: True if path is allowed, False otherwise
    """
    if ALLOW_SYSTEM_ACCESS:
        return True

    return str(path).startswith(str(WORKING_DIR))


def get_python_environments() -> List[Dict[str, str]]:
    """Get only the currently active Python environment."""
    python_path = args.python_path if args.python_path else sys.executable
    return {
        "path": python_path,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def get_installed_packages(python_path: str) -> List[Dict[str, str]]:
    """Get installed packages for a specific Python environment."""
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting installed packages: {e}")
        return []


def execute_python_code(code: str, python_path: str | None = None, working_dir: str | None = None) -> Dict[str, Any]:
    """
    Execute Python code and return the result.

    Args:
        code: Python code to execute
        python_path: Path to Python executable (default: custom or system Python)
        working_dir: Working directory for execution

    Returns:
        Dict with stdout, stderr, and status
    """
    if python_path is None:
        python_path = DEFAULT_PYTHON_PATH

    # Combine session functions and user code
    combined_code = ""
    if SESSION_FUNCTIONS_FILE.exists():
        with open(SESSION_FUNCTIONS_FILE) as f:
            combined_code += f.read() + "\n"
    combined_code += code

    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(combined_code)
        temp_path = temp.name

    try:
        result = subprocess.run([python_path, temp_path], capture_output=True, text=True, cwd=working_dir)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": result.returncode,
        }
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def find_python_files(directory: str | Path) -> List[Dict[str, str]]:
    """Find all Python files in a directory and its subdirectories."""
    files = []

    directory_path = Path(directory)
    if not directory_path.exists():
        return files  # Return empty list instead of throwing error

    for path in directory_path.rglob("*.py"):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                }
            )

    return files


@mcp.resource("python://environments")
def get_environments_resource() -> str:
    """List all available Python environments as a resource."""
    environments = get_python_environments()
    return json.dumps(environments, indent=2)


@mcp.resource("python://packages")
def get_packages_resource() -> str:
    """List installed packages for a specific environment as a resource."""
    env = get_python_environments()
    packages = get_installed_packages(env["path"])
    return json.dumps(packages, indent=2)


@mcp.tool()
def read_file(file_path: str, max_size_kb: int = 1024) -> str:
    """
    Read the content of any file, with size limits for safety.

    Args:
        file_path: Path to the file (relative to working directory or absolute)
        max_size_kb: Maximum file size to read in KB (default: 1024)

    Returns:
        str: File content or an error message
    """
    # Handle path based on security settings
    path = Path(file_path)
    if path.is_absolute():
        if not is_path_allowed(path):
            return f"Access denied: System-wide file access is {'DISABLED' if not ALLOW_SYSTEM_ACCESS else 'ENABLED, but this path is not allowed'}"
    else:
        # Make path relative to working directory if it's not already absolute
        path = WORKING_DIR / path

    try:
        if not path.exists():
            return f"Error: File '{file_path}' not found"

        # Check file size
        file_size_kb = path.stat().st_size / 1024
        if file_size_kb > max_size_kb:
            return f"Error: File size ({file_size_kb:.2f} KB) exceeds maximum allowed size ({max_size_kb} KB)"

        # Determine file type and read accordingly
        try:
            # Try to read as text first
            with open(path, encoding="utf-8") as f:
                content = f.read()

            # If it's a known source code type, use code block formatting
            source_code_extensions = [
                ".py",
                ".js",
                ".html",
                ".css",
                ".json",
                ".xml",
                ".md",
                ".txt",
                ".sh",
                ".c",
                ".cpp",
                ".java",
                ".rb",
            ]
            if path.suffix.lower() in source_code_extensions:
                file_type = path.suffix[1:] if path.suffix else "plain"
                return f"File: {file_path}\n\n```{file_type}\n{content}\n```"

            # For other text files, return as-is
            return f"File: {file_path}\n\n{content}"

        except UnicodeDecodeError:
            # If text decoding fails, read as binary and show hex representation
            with open(path, "rb") as f:
                content = f.read()
                hex_content = content.hex()
                return f"Binary file: {file_path}\nFile size: {len(content)} bytes\nHex representation (first 1024 chars):\n{hex_content[:1024]}"

    except Exception as e:
        return f"Error reading file {file_path}: {e!s}"


@mcp.tool()
def write_file(file_path: str, content: str, overwrite: bool = False, encoding: str = "utf-8") -> str:
    """
    Write content to a file in the working directory or system-wide if allowed.

    Args:
        file_path: Path to the file to write (relative to working directory or absolute if system access is enabled)
        content: Content to write to the file
        overwrite: Whether to overwrite the file if it exists (default: False)
        encoding: File encoding (default: utf-8)

    Returns:
        str: Status message about the file writing operation
    """
    # Handle path based on security settings
    path = Path(file_path)
    if path.is_absolute():
        if not is_path_allowed(path):
            return f"For security reasons, you can only write files inside the working directory: {WORKING_DIR} (System-wide access is disabled)"
    else:
        # Make path relative to working directory if it's not already
        path = WORKING_DIR / path

    try:
        # Check if the file exists
        if path.exists() and not overwrite:
            return f"File '{path}' already exists. Use overwrite=True to replace it."

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine write mode based on content type
        if isinstance(content, str):
            # Text content
            with open(path, "w", encoding=encoding) as f:
                f.write(content)
        elif isinstance(content, bytes):
            # Binary content
            with open(path, "wb") as f:
                f.write(content)
        else:
            return f"Unsupported content type: {type(content)}"

        # Get file information
        file_size_kb = path.stat().st_size / 1024
        return f"Successfully wrote to {path}. File size: {file_size_kb:.2f} KB"

    except Exception as e:
        return f"Error writing to file: {e!s}"


@mcp.resource("python://directory")
def get_working_directory_listing() -> str:
    """List all Python files in the working directory as a resource."""
    try:
        files = find_python_files(WORKING_DIR)
        return json.dumps({"working_directory": str(WORKING_DIR), "files": files}, indent=2)
    except Exception as e:
        return f"Error listing directory: {e!s}"


@mcp.tool()
def list_directory(directory_path: str = "") -> str:
    """
    List all Python files in a directory or subdirectory.

    Args:
        directory_path: Path to directory (relative to working directory or absolute, empty for working directory)
    """
    try:
        # Handle empty path (use working directory)
        if not directory_path:
            path = WORKING_DIR
        else:
            # Handle absolute paths
            path = Path(directory_path)
            if path.is_absolute():
                if not is_path_allowed(path):
                    return f"Access denied: System-wide file access is {'DISABLED' if not ALLOW_SYSTEM_ACCESS else 'ENABLED, but this path is not allowed'}"
            else:
                # Make path relative to working directory if it's not already absolute
                path = WORKING_DIR / directory_path

        # Check if directory exists
        if not path.exists():
            return f"Error: Directory '{directory_path}' not found"

        if not path.is_dir():
            return f"Error: '{directory_path}' is not a directory"

        files = find_python_files(path)

        if not files:
            return f"No Python files found in {directory_path or 'working directory'}"

        result = f"Python files in directory: {directory_path or str(WORKING_DIR)}\n\n"

        # Group files by subdirectory for better organization
        files_by_dir = {}
        base_dir = path if ALLOW_SYSTEM_ACCESS else WORKING_DIR

        for file in files:
            file_path = Path(file["path"])
            try:
                relative_path = file_path.relative_to(base_dir)
                parent = str(relative_path.parent)

                if parent == ".":
                    parent = "(root)"
            except ValueError:
                # This can happen with system-wide access enabled
                parent = str(file_path.parent)

            if parent not in files_by_dir:
                files_by_dir[parent] = []

            files_by_dir[parent].append(
                {
                    "name": file["name"],
                    "size": file["size"],
                    "modified": file["modified"],
                }
            )

        # Format the output
        for dir_name, dir_files in sorted(files_by_dir.items()):
            result += f"📁 {dir_name}:\n"
            for file in sorted(dir_files, key=lambda x: x["name"]):
                size_kb = round(file["size"] / 1024, 1)
                result += f"  📄 {file['name']} ({size_kb} KB)\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error listing directory: {e!s}"


# ============================================================================
# Tools
# ============================================================================


@mcp.tool()
def list_python_environments() -> str:
    """List all available Python environments (system Python and conda environments)."""
    environments = get_python_environments()

    if not environments:
        return "No Python environments found."

    result = "Available Python Environments:\n\n"
    for env in environments:
        result += f"- Name: {env['name']}\n"
        result += f"  Path: {env['path']}\n"
        result += f"  Version: Python {env['version']}\n\n"

    return result


@mcp.tool()
def list_installed_packages(environment: str = "default") -> str:
    """
    List installed packages for a specific Python environment.

    Args:
        environment: Name of the Python environment (default: default if custom path provided, otherwise system)
    """
    env = get_python_environments()

    packages = get_installed_packages(env["path"])

    if not packages:
        return f"No packages found in environment '{environment}'."

    result = f"Installed Packages in '{environment}':\n\n"
    for pkg in packages:
        result += f"- {pkg['name']} {pkg['version']}\n"

    return result


@mcp.tool()
def run_python_code(code: str, environment: str = "default", save_as: str | None = None) -> str:
    """
    Execute Python code and return the result. Code runs in the working directory.

    Args:
        code: Python code to execute
        environment: Name of the Python environment to use (default if custom path provided, otherwise system)
        save_as: Optional filename to save the code before execution (useful for future reference)
    """
    env = get_python_environments()

    # Optionally save the code to a file
    if save_as:
        save_path = WORKING_DIR / save_as

        # Ensure filename has .py extension
        if not save_path.suffix == ".py":
            save_path = save_path.with_suffix(".py")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(code)
        except Exception as e:
            return f"Error saving code to file: {e!s}"

    # Execute the code
    result = execute_python_code(code, env["path"], WORKING_DIR)

    output = f"Execution in '{environment}' environment"
    if save_as:
        output += f" (saved to {save_as})"
    output += ":\n\n"

    if result["status"] == 0:
        output += "--- Output ---\n"
        if result["stdout"]:
            output += result["stdout"]
        else:
            output += "(No output)\n"
    else:
        output += f"--- Error (status code: {result['status']}) ---\n"
        if result["stderr"]:
            output += result["stderr"]
        else:
            output += "(No error message)\n"

        if result["stdout"]:
            output += "\n--- Output ---\n"
            output += result["stdout"]

    return output


@mcp.tool()
def install_package(package_name: str, upgrade: bool = False) -> str:
    """
    Install a Python package in the specified environment.

    Args:
        package_name: Name of the package to install
        environment: Name of the Python environment (default if custom path provided, otherwise system)
        upgrade: Whether to upgrade the package if already installed (default: False)
    """
    env = get_python_environments()

    # Build the pip command
    cmd = [env["path"], "-m", "pip", "install"]

    if upgrade:
        cmd.append("--upgrade")

    cmd.append(package_name)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            return f"Successfully {'upgraded' if upgrade else 'installed'} {package_name}."
        else:
            return f"Error installing {package_name}:\n{result.stderr}"
    except Exception as e:
        return f"Error installing package: {e!s}"


@mcp.tool()
def write_python_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Write content to a Python file in the working directory or system-wide if allowed.

    Args:
        file_path: Path to the file to write (relative to working directory or absolute if system access is enabled)
        content: Content to write to the file
        overwrite: Whether to overwrite the file if it exists (default: False)
    """
    # Handle path based on security settings
    path = Path(file_path)
    if path.is_absolute():
        if not is_path_allowed(path):
            security_status = "DISABLED" if not ALLOW_SYSTEM_ACCESS else "ENABLED, but this path is not allowed"
            return f"For security reasons, you can only write files inside the working directory: {WORKING_DIR} (System-wide access is {security_status})"
    else:
        # Make path relative to working directory if it's not already
        path = WORKING_DIR / path

    # Check if the file exists
    if path.exists() and not overwrite:
        return f"File '{path}' already exists. Use overwrite=True to replace it."

    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(path, "w") as f:
            f.write(content)

        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing to file: {e!s}"


@mcp.tool()
def run_python_file(file_path: str, arguments: List[str] | None = None) -> str:
    """
    Execute a Python file and return the result.

    Args:
        file_path: Path to the python file to execute (relative to working directory or absolute if the system access
            is enabled)
        arguments: List of command-line arguments to pass to the script
    """
    # Handle path based on security settings
    path = Path(file_path)
    if path.is_absolute():
        if not is_path_allowed(path):
            return (
                f"For security reasons, you can only run files inside the working directory: {WORKING_DIR} "
                "unless the system access is enabled"
            )
    else:
        # Make path relative to working directory if it's not already
        path = WORKING_DIR / path

    if not path.exists():
        return f"File '{path}' not found."

    env = get_python_environments()

    # Build the command
    cmd = [env["path"], str(path)]

    if arguments:
        cmd.extend(arguments)

    try:
        # Run the command with working directory set properly
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=WORKING_DIR)

        output = f"Execution of '{path}':\n\n"

        if result.returncode == 0:
            output += "--- Output ---\n"
            if result.stdout:
                output += result.stdout
            else:
                output += "(No output)\n"
        else:
            output += f"--- Error (status code: {result.returncode}) ---\n"
            if result.stderr:
                output += result.stderr
            else:
                output += "(No error message)\n"

            if result.stdout:
                output += "\n--- Output ---\n"
                output += result.stdout

        return output
    except Exception as e:
        return f"Error executing file: {e!s}"


@mcp.tool()
def register_functions(code: str) -> str:
    """
    Register python code that define functions to the session file, so that they can be called later.
    """
    try:
        # Append code to session file
        with open(SESSION_FUNCTIONS_FILE, "a") as f:
            f.write("\n" + code + "\n")
        return "Function code registered to session file."
    except Exception as e:
        return f"Error registering functions: {e}"


@mcp.tool()
def cleanup():
    _cleanup_session_file()
    return "cleanup done."


# Run the server when executed directly
if __name__ == "__main__":
    mcp.run()
