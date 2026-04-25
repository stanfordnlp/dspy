#!/usr/bin/env python3
"""
DSPy ReAct SweBench Benchmarking Script

This script benchmarks DSPy ReAct's performance on the SweBench benchmark.
Input: Language model name (e.g., 'openai/gpt-4o-mini')
Output: Overall SweBench score

Usage:
    python scripts/benchmark_swebench.py --model openai/gpt-4o-mini
    python scripts/benchmark_swebench.py --model openai/gpt-4o-mini --dataset swe-bench-lite --max-instances 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add parent directory to path to import dspy
sys.path.insert(0, str(Path(__file__).parent.parent))

from swebench_adapter import SWeBenchAdapter
from swebench_tools import get_swebench_tools

import dspy


class SWeBenchBenchmarker:
    """Main benchmarking class for DSPy ReAct on SweBench."""

    def __init__(
        self,
        model_name: str,
        dataset: str = "swe-bench-lite",
        max_instances: int | None = None,
        output_dir: str = "swebench_results",
        max_workers: int = 4,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.max_instances = max_instances
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.adapter = SWeBenchAdapter()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DSPy LM
        self.lm = self._initialize_lm(model_name)
        dspy.settings.configure(lm=self.lm)

    def _initialize_lm(self, model_name: str) -> dspy.LM:
        """Initialize DSPy language model from model name.

        Args:
            model_name: Model name (e.g., 'openai/gpt-4o-mini')

        Returns:
            Configured DSPy language model
        """
        print(f"Initializing language model: {model_name}")

        if model_name.startswith("openai/"):
            model_id = model_name.replace("openai/", "")
            return dspy.LM(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name.startswith("anthropic/"):
            model_id = model_name.replace("anthropic/", "")
            return dspy.LM(model=model_id, api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            # Try generic initialization
            return dspy.LM(model=model_name)

    def _install_swebench(self) -> None:
        """Install SweBench package if not already installed."""
        try:
            import swebench

            print("SweBench already installed")
        except ImportError:
            print("Installing SweBench...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "swebench"])

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load SweBench dataset instances.

        Returns:
            List of SweBench instances
        """
        print(f"Loading {self.dataset} dataset...")

        # Map dataset names to actual dataset identifiers
        dataset_map = {
            "swe-bench": "princeton-nlp/SWE-bench",
            "swe-bench-lite": "princeton-nlp/SWE-bench_Lite",
            "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
        }

        dataset_name = dataset_map.get(self.dataset, self.dataset)

        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_name, split="test")
            instances = list(dataset)

            if self.max_instances:
                instances = instances[: self.max_instances]

            print(f"Loaded {len(instances)} instances from {dataset_name}")
            return instances

        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback: try to load from local file if provided
            if os.path.exists(self.dataset):
                return self.adapter.load_swebench_instances(self.dataset)
            else:
                raise

    def _setup_repository(self, instance: dict[str, Any]) -> Path:
        """Set up repository for a SweBench instance.

        Args:
            instance: SweBench instance

        Returns:
            Path to the repository
        """
        repo_path = self.output_dir / "repos" / instance["instance_id"]
        repo_path.mkdir(parents=True, exist_ok=True)

        # Clone repository if not exists
        if not (repo_path / ".git").exists():
            repo_url = f"https://github.com/{instance['repo']}.git"
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True, capture_output=True)

        # Checkout the correct commit
        subprocess.run(["git", "checkout", instance["base_commit"]], cwd=repo_path, check=True, capture_output=True)

        return repo_path

    def _create_react_agent(self, repo_path: Path) -> dspy.ReAct:
        """Create DSPy ReAct agent with SweBench tools.

        Args:
            repo_path: Path to the repository

        Returns:
            Configured ReAct agent
        """
        tools = get_swebench_tools(work_dir=str(repo_path))
        signature = self.adapter.create_react_signature()

        react_agent = dspy.ReAct(signature=signature, tools=tools, max_iters=10)

        return react_agent

    def _process_single_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Process a single SweBench instance.

        Args:
            instance: SweBench instance

        Returns:
            Prediction result
        """
        instance_id = instance["instance_id"]
        print(f"Processing {instance_id}...")

        try:
            # Set up repository
            repo_path = self._setup_repository(instance)

            # Create ReAct agent
            react_agent = self._create_react_agent(repo_path)

            # Format input for ReAct
            react_input = self.adapter.format_input_for_react(instance)

            # Run ReAct agent
            prediction = react_agent(**react_input)

            import pdb

            pdb.set_trace()

            # Format prediction for SweBench
            formatted_prediction = self.adapter.format_prediction(
                instance_id=instance_id, react_prediction=prediction, repo_path=str(repo_path)
            )

            return formatted_prediction

        except Exception as e:
            print(f"Error processing {instance_id}: {e}")
            return {"instance_id": instance_id, "model_patch": f"# Error: {e!s}", "model_name_or_path": "dspy_react"}

    def run_benchmark(self) -> float:
        """Run the complete SweBench benchmark.

        Returns:
            Overall SweBench score
        """
        print(f"Starting DSPy ReAct SweBench benchmark with model: {self.model_name}")

        # Install SweBench
        self._install_swebench()

        # Load dataset
        instances = self._load_dataset()

        # Process instances
        predictions = []
        for instance in tqdm(instances, desc="Processing instances"):
            prediction = self._process_single_instance(instance)
            predictions.append(prediction)
            break

        # Save predictions
        predictions_file = self.output_dir / "predictions.jsonl"
        self.adapter.save_predictions(predictions, str(predictions_file))

        # Run SweBench evaluation
        score = self._run_swebench_evaluation(predictions_file)

        print(f"SweBench Score: {score}")
        return score

    def _run_swebench_evaluation(self, predictions_file: Path) -> float:
        """Run SweBench evaluation on predictions.

        Args:
            predictions_file: Path to predictions file

        Returns:
            SweBench score
        """
        print("Running SweBench evaluation...")

        try:
            # Map dataset names for evaluation
            dataset_map = {
                "swe-bench": "princeton-nlp/SWE-bench",
                "swe-bench-lite": "princeton-nlp/SWE-bench_Lite",
                "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
            }

            dataset_name = dataset_map.get(self.dataset, self.dataset)

            # Run evaluation using SweBench harness
            cmd = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset_name,
                "--predictions_path",
                str(predictions_file.absolute()),  # Use absolute path
                "--max_workers",
                str(self.max_workers),
                "--run_id",
                f"dspy_react_{int(time.time())}",
            ]

            result = subprocess.run(cmd, cwd=self.output_dir, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"SweBench evaluation failed: {result.stderr}")
                return 0.0

            # Parse results
            score = self._parse_evaluation_results()
            return score

        except Exception as e:
            print(f"Error running SweBench evaluation: {e}")
            return 0.0

    def _parse_evaluation_results(self) -> float:
        """Parse SweBench evaluation results to extract score.

        Returns:
            Overall score
        """
        try:
            # Look for results in the evaluation output
            results_dir = self.output_dir / "evaluation_results"

            if results_dir.exists():
                # Find the most recent results file
                result_files = list(results_dir.glob("*.json"))
                if result_files:
                    with open(result_files[0]) as f:
                        results = json.load(f)

                    # Extract score (typically resolve_rate)
                    if "resolve_rate" in results:
                        return results["resolve_rate"]
                    elif "score" in results:
                        return results["score"]

            # Fallback: simple success counting
            predictions_file = self.output_dir / "predictions.jsonl"
            if predictions_file.exists():
                total = 0
                success = 0

                with open(predictions_file) as f:
                    for line in f:
                        if line.strip():
                            total += 1
                            prediction = json.loads(line)
                            # Simple heuristic: non-empty patch indicates attempt
                            if prediction.get("model_patch", "").strip():
                                success += 1

                return success / total if total > 0 else 0.0

            return 0.0

        except Exception as e:
            print(f"Error parsing evaluation results: {e}")
            return 0.0


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(description="DSPy ReAct SweBench Benchmark")
    parser.add_argument("--model", required=True, help="Language model name (e.g., openai/gpt-4o-mini)")
    parser.add_argument(
        "--dataset",
        default="swe-bench-lite",
        choices=["swe-bench", "swe-bench-lite", "swe-bench-verified"],
        help="SweBench dataset variant to use",
    )
    parser.add_argument("--max-instances", type=int, help="Maximum number of instances to process (for testing)")
    parser.add_argument("--output-dir", default="swebench_results", help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes for evaluation")

    args = parser.parse_args()

    # Check Docker availability
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Docker is required for SweBench evaluation but is not available.")
        print("Please install Docker and ensure it's running.")
        sys.exit(1)

    # Run benchmark
    benchmarker = SWeBenchBenchmarker(
        model_name=args.model,
        dataset=args.dataset,
        max_instances=args.max_instances,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )

    try:
        score = benchmarker.run_benchmark()
        print(f"\nFinal SweBench Score: {score:.4f}")

        # Save summary
        summary = {"model": args.model, "dataset": args.dataset, "score": score, "timestamp": time.time()}

        summary_file = benchmarker.output_dir / "benchmark_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {benchmarker.output_dir}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
