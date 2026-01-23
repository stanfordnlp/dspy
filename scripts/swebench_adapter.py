#!/usr/bin/env python3
"""Adapter to bridge DSPy ReAct outputs with SweBench format requirements."""

import json
import re
from typing import Any

import dspy


class SWeBenchAdapter:
    """Converts DSPy ReAct predictions to SweBench format."""

    def __init__(self):
        self.instance_predictions = []

    def format_prediction(self, instance_id: str, react_prediction: Any, repo_path: str) -> dict[str, Any]:
        """Format a single DSPy ReAct prediction for SweBench evaluation.

        Args:
            instance_id: SweBench instance ID
            react_prediction: DSPy ReAct prediction object
            repo_path: Path to the repository

        Returns:
            Formatted prediction dictionary
        """
        # Extract the model patch from ReAct prediction
        model_patch = self._extract_patch_from_prediction(react_prediction, repo_path)

        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": "dspy_react",
        }

        return prediction

    def _extract_patch_from_prediction(self, prediction: Any, repo_path: str) -> str:
        """Extract unified diff patch from DSPy ReAct prediction.

        Args:
            prediction: DSPy ReAct prediction object
            repo_path: Path to the repository

        Returns:
            Unified diff patch string
        """
        try:
            # First check the main solution output
            if hasattr(prediction, "solution"):
                solution = prediction.solution
                if isinstance(solution, str) and self._looks_like_git_diff(solution):
                    return solution

            # Check if there's a trajectory with git diff information
            if hasattr(prediction, "trajectory") and prediction.trajectory:
                trajectory = prediction.trajectory

                # Look specifically for generate_final_patch outputs
                for key, value in trajectory.items():
                    if isinstance(value, str):
                        # Prioritize generate_final_patch tool output
                        if "generate_final_patch" in key.lower() and self._looks_like_git_diff(value):
                            return value
                        
                        # Check for any git diff output
                        if self._looks_like_git_diff(value):
                            return value

                        # Check for patch content in tool outputs
                        if "diff --git" in value or "--- a/" in value:
                            return value

            # Check other possible attributes
            for attr in ["answer", "output", "result"]:
                if hasattr(prediction, attr):
                    value = getattr(prediction, attr)
                    if isinstance(value, str) and self._looks_like_git_diff(value):
                        return value

            # If no explicit patch found, try to generate one from file changes
            return self._generate_patch_from_changes(prediction, repo_path)

        except Exception as e:
            print(f"Error extracting patch: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _looks_like_git_diff(self, text: str) -> bool:
        """Check if text looks like a git diff output.

        Args:
            text: Text to check

        Returns:
            True if text appears to be a git diff
        """
        diff_indicators = [
            "diff --git",
            "--- a/",
            "+++ b/",
            "@@ -",
            "index ",
        ]

        return any(indicator in text for indicator in diff_indicators)

    def _generate_patch_from_changes(self, prediction: Any, repo_path: str) -> str:
        """Generate a patch from file changes mentioned in the prediction.

        Args:
            prediction: DSPy ReAct prediction object
            repo_path: Path to the repository

        Returns:
            Generated patch string
        """
        try:
            # This is a fallback method - in practice, the ReAct agent should
            # generate explicit git diff output through the tools

            # Try to extract file modifications from the trajectory
            if hasattr(prediction, "trajectory") and prediction.trajectory:
                trajectory = prediction.trajectory

                # Look for write_file operations
                written_files = []
                for key, value in trajectory.items():
                    if "write_file" in key.lower() or "Successfully wrote" in str(value):
                        # Extract file path from the operation
                        file_match = re.search(r"([^\s]+\.py)", str(value))
                        if file_match:
                            written_files.append(file_match.group(1))

                if written_files:
                    # Generate a simple indication that files were modified
                    patch_lines = []
                    for file_path in written_files:
                        patch_lines.extend(
                            [
                                f"diff --git a/{file_path} b/{file_path}",
                                "index 0000000..1111111 100644",
                                f"--- a/{file_path}",
                                f"+++ b/{file_path}",
                                "@@ -1,1 +1,1 @@",
                                "# File modified by DSPy ReAct agent",
                            ]
                        )

                    return "\n".join(patch_lines)

            return "# No patch generated - check ReAct trajectory for modifications"

        except Exception as e:
            print(f"Error generating patch: {e}")
            return ""

    def save_predictions(self, predictions: list[dict[str, Any]], output_file: str) -> None:
        """Save predictions to JSON Lines file for SweBench evaluation.

        Args:
            predictions: List of formatted predictions
            output_file: Path to output file
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for prediction in predictions:
                    f.write(json.dumps(prediction) + "\n")

            print(f"Saved {len(predictions)} predictions to {output_file}")

        except Exception as e:
            print(f"Error saving predictions: {e}")
            raise

    def load_swebench_instances(self, dataset_file: str) -> list[dict[str, Any]]:
        """Load SweBench instances from dataset file.

        Args:
            dataset_file: Path to SweBench dataset file

        Returns:
            List of SweBench instances
        """
        try:
            instances = []

            if dataset_file.endswith(".jsonl"):
                with open(dataset_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            instances.append(json.loads(line))
            else:
                with open(dataset_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        instances = data
                    else:
                        instances = [data]

            return instances

        except Exception as e:
            print(f"Error loading SweBench instances: {e}")
            raise

    def create_react_signature(self) -> str:
        """Create DSPy signature for SweBench code generation task.

        Returns:
            DSPy signature string
        """

        class MySignature(dspy.Signature):
            """Given a problem statement and repository structure, analyze the issue and generate a solution.
            
            CRITICAL: You MUST use the 'generate_final_patch' tool at the end to create a git diff patch.
            Your solution should include:
            1. Understanding of the problem
            2. Identification of relevant files using list_files and search_code
            3. Making necessary changes using write_file
            4. Testing your changes using run_tests if possible
            5. MOST IMPORTANT: Use generate_final_patch to create the final git diff
            
            The solution output should contain the actual git diff patch from generate_final_patch tool."""

            problem_statement: str = dspy.InputField()
            repo_structure: str = dspy.InputField()
            solution: str = dspy.OutputField(desc="Must contain the final git diff patch from generate_final_patch tool")

        return MySignature

    def format_input_for_react(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Format SweBench instance for DSPy ReAct input.

        Args:
            instance: SweBench instance dictionary

        Returns:
            Formatted input for ReAct
        """
        problem_statement = instance.get("problem_statement", "")

        # Include additional context if available
        if "hint_text" in instance:
            problem_statement += f"\n\nHint: {instance['hint_text']}"

        repo_structure = "Repository structure will be explored using tools."

        return {"problem_statement": problem_statement, "repo_structure": repo_structure}
