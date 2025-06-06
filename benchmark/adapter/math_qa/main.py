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


def print_results(model, results):
    name_mapping = {
        "vanilla_sdk": "Vanilla",
        "vanilla_sdk_with_structured_output": "Vanilla with Structured Output",
        "dspy_cot_chat_adapter": "DSPy (ChatAdapter)",
        "dspy_cot_json_adapter": "DSPy (JSONAdapter)",
    }

    headers = [name_mapping[k] for k in results.keys()]
    score_row = []
    time_row = []
    for v in results.values():
        # v can be a dict with 'score' or 'scores', and 'time_taken'
        score = v.get("score")
        if score is None:
            score = v.get("scores")
            if isinstance(score, list):
                score = f"{sum(score) / len(score):.3f}" if score else "-"
        score_row.append(str(score))
        time_row.append(f"{v['time_taken']:.2f}s")

    table = [score_row, time_row]
    row_labels = ["Score", "Time Cost"]
    table_with_labels = [[label] + row for label, row in zip(row_labels, table)]

    print(f"\nBenchmark Results Summary for {model}:")
    print(tabulate(table_with_labels, headers=[""] + headers, tablefmt="github"))


def main():
    parser = argparse.ArgumentParser(description="Run math QA benchmarking with different LMs.")
    parser.add_argument("--model", type=str, required=True, help="The model name of the LM to use.")
    parser.add_argument("--num_threads", type=int, default=1, help="The number of threads to use for the benchmark.")
    args = parser.parse_args()

    # Disable cache for fair comparison.
    dspy.configure_cache(enable_disk_cache=False, enable_litellm_cache=False)

    dataset = MATH(subset="algebra").dev[:10]
    model = args.model

    final_result = {}

    final_result["vanilla_sdk"] = vanilla_sdk(dataset, model, args.num_threads)
    final_result["vanilla_sdk_with_structured_output"] = vanilla_sdk_with_structured_output(
        dataset, model, args.num_threads
    )
    final_result["dspy_cot_chat_adapter"] = dspy_cot_chat_adapter(dataset, model, args.num_threads)
    final_result["dspy_cot_json_adapter"] = dspy_cot_json_adapter(dataset, model, args.num_threads)

    print_results(model, final_result)


if __name__ == "__main__":
    main()
