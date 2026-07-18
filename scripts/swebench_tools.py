#!/usr/bin/env python3
"""SweBench-specific tools for DSPy ReAct agent."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class SWeBenchTools:
    """Tools for SweBench code generation and editing tasks."""
    
    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """Read the contents of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File contents as string
        """
        try:
            full_path = self.work_dir / file_path
            if not full_path.exists():
                return f"Error: File {file_path} does not exist"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Limit content size for context window
            if len(content) > 10000:
                content = content[:10000] + f"\n... (truncated, total length: {len(content)} chars)"
            
            return content
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            Success or error message
        """
        try:
            full_path = self.work_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing file {file_path}: {str(e)}"
    
    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """List files in a directory.
        
        Args:
            directory: Directory to list files in
            pattern: Glob pattern to match files
            
        Returns:
            List of files as string
        """
        try:
            full_path = self.work_dir / directory
            if not full_path.exists():
                return f"Error: Directory {directory} does not exist"
            
            files = list(full_path.glob(pattern))
            files = [f.relative_to(self.work_dir) for f in files if f.is_file()]
            files.sort()
            
            if len(files) > 50:
                file_list = '\n'.join(str(f) for f in files[:50])
                file_list += f"\n... (showing first 50 of {len(files)} files)"
            else:
                file_list = '\n'.join(str(f) for f in files)
            
            return file_list
        except Exception as e:
            return f"Error listing files in {directory}: {str(e)}"
    
    def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a shell command.
        
        Args:
            command: Command to run
            timeout: Timeout in seconds
            
        Returns:
            Command output or error message
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            if result.returncode != 0:
                output += f"\nCommand failed with return code {result.returncode}"
            
            # Limit output size
            if len(output) > 5000:
                output = output[:5000] + f"\n... (truncated, total length: {len(output)} chars)"
            
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"
    
    def run_tests(self, test_command: str = None) -> str:
        """Run tests in the repository.
        
        Args:
            test_command: Custom test command, defaults to common test runners
            
        Returns:
            Test results
        """
        if test_command:
            return self.run_command(test_command)
        
        # Try common test commands
        test_commands = [
            "python -m pytest",
            "python -m unittest discover",
            "python test.py",
            "python tests.py",
            "make test",
            "npm test",
            "cargo test"
        ]
        
        for cmd in test_commands:
            result = self.run_command(f"which {cmd.split()[0]}")
            if "not found" not in result.lower():
                return self.run_command(cmd)
        
        return "No suitable test command found. Please specify test_command."
    
    def search_code(self, pattern: str, file_pattern: str = "*.py") -> str:
        """Search for code patterns in files.
        
        Args:
            pattern: Pattern to search for
            file_pattern: File pattern to search in
            
        Returns:
            Search results
        """
        try:
            # Use grep for code search
            cmd = f"grep -r --include='{file_pattern}' -n '{pattern}' ."
            result = self.run_command(cmd)
            
            if "No such file or directory" in result or not result.strip():
                return f"No matches found for pattern '{pattern}' in {file_pattern}"
            
            lines = result.split('\n')
            if len(lines) > 20:
                result = '\n'.join(lines[:20]) + f"\n... (showing first 20 of {len(lines)} matches)"
            
            return result
        except Exception as e:
            return f"Error searching code: {str(e)}"
    
    def get_git_diff(self) -> str:
        """Get current git diff.
        
        Returns:
            Git diff output
        """
        diff_output = self.run_command("git diff")
        
        # If no changes are staged, try git diff --cached and git status
        if not diff_output.strip() or "no changes added to commit" in diff_output.lower():
            cached_diff = self.run_command("git diff --cached")
            if cached_diff.strip():
                return cached_diff
            
            # Also try git diff HEAD to see all changes
            head_diff = self.run_command("git diff HEAD")
            if head_diff.strip():
                return head_diff
                
        return diff_output
    
    def apply_patch(self, patch_content: str) -> str:
        """Apply a patch to the repository.
        
        Args:
            patch_content: Patch content in unified diff format
            
        Returns:
            Result of patch application
        """
        try:
            # Write patch to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch_content)
                patch_file = f.name
            
            # Apply the patch
            result = self.run_command(f"git apply {patch_file}")
            
            # Clean up
            os.unlink(patch_file)
            
            return result
        except Exception as e:
            return f"Error applying patch: {str(e)}"
    
    def generate_final_patch(self) -> str:
        """Generate final patch for all changes made during the session.
        
        This should be called at the end to capture all modifications.
        
        Returns:
            Complete git diff patch
        """
        # First add all changes to git tracking
        self.run_command("git add .")
        
        # Generate comprehensive diff
        diff_output = self.run_command("git diff --cached")
        
        if not diff_output.strip():
            # Fallback to unstaged changes
            diff_output = self.run_command("git diff")
            
        if not diff_output.strip():
            # Last resort: check git status and create minimal diff
            status = self.run_command("git status --porcelain")
            if status.strip():
                return f"# Changes detected but no diff generated:\n{status}\n# Use git diff to see actual changes"
            else:
                return "# No changes detected"
                
        return diff_output


def get_swebench_tools(work_dir: Optional[str] = None) -> List[callable]:
    """Get list of SweBench tools for DSPy ReAct.
    
    Args:
        work_dir: Working directory for the tools
        
    Returns:
        List of tool functions
    """
    tools_instance = SWeBenchTools(work_dir)
    
    return [
        tools_instance.read_file,
        tools_instance.write_file,
        tools_instance.list_files,
        tools_instance.run_command,
        tools_instance.run_tests,
        tools_instance.search_code,
        tools_instance.get_git_diff,
        tools_instance.apply_patch,
        tools_instance.generate_final_patch,
    ]