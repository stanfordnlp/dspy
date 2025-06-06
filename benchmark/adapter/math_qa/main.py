import argparse

from tabulate import tabulate

import dspy
from dspy.datasets import MATH

from .program import (
    dspy_cot_chat_adapter,
    dspy_cot_json_adapter,
    vanilla_sdk,
    vanilla_sdk_with_structured_output,
)


def print_results(results):
    program_name_mapping = {
        "vanilla_sdk": "Vanilla",
        "vanilla_sdk_with_structured_output": "Vanilla with Structured Output",
        "dspy_cot_chat_adapter": "DSPy (ChatAdapter)",
        "dspy_cot_json_adapter": "DSPy (JSONAdapter)",
    }

    headers = list(program_name_mapping.values())

    table = []
    for model_name, result in results.items():
        score_row = [model_name]
        for program_name in program_name_mapping.keys():
            score = result[program_name].get("score", 0.0)
            score_row.append(f"{score:.2f}")

        table.append(score_row)

    print("\nBenchmark Results Summary:\n")

    print(tabulate(table, headers=[""] + headers, tablefmt="github"))


def main():
    parser = argparse.ArgumentParser(description="Run math QA benchmarking with different LMs.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["openai/gpt-4o-mini", "openai/gpt-4.1", "anthropic/claude-3.5-sonnet"],
        help="The model name(s) of the LM(s) to use. (default: %(default)s)",
    )
    parser.add_argument("--num_threads", type=int, default=1, help="The number of threads to use for the benchmark.")
    parser.add_argument("--enable_cache", action="store_true", help="Enable cache for the benchmark.")
    args = parser.parse_args()

    # If we enable cache, we always use litellm cache for fair comparison.
    dspy.configure_cache(enable_disk_cache=False, enable_litellm_cache=args.enable_cache)

    dataset = MATH(subset="algebra").dev[:10]

    final_results = {}
    for model in args.models:
        model_results = {}
        model_results["vanilla_sdk"] = vanilla_sdk(dataset, model, args.num_threads)
        model_results["vanilla_sdk_with_structured_output"] = vanilla_sdk_with_structured_output(
            dataset, model, args.num_threads
        )
        model_results["dspy_cot_chat_adapter"] = dspy_cot_chat_adapter(dataset, model, args.num_threads)
        model_results["dspy_cot_json_adapter"] = dspy_cot_json_adapter(dataset, model, args.num_threads)

        final_results[model] = model_results

    print_results(final_results)


if __name__ == "__main__":
    main()
