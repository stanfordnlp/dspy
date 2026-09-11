"""Run with Python from a clean DSPy 3.3.0 checkout, not the migration branch.

Usage: python /path/to/generate.py OUTPUT_DIRECTORY
The subprocess working directory must be the 3.3.0 checkout. No network calls
are made; only LiteLLM's response constructors and DSPy's actual cache are used.
"""

import asyncio
import copy
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

SOURCE_COMMIT = "e4e97aae29b8ad8aa2fb7e99ffae6fd52970fad8"
HISTORY_KEYS = ("prompt", "messages", "kwargs", "outputs", "usage", "cost", "model", "response_model", "model_type")


def plain(value):
    return json.loads(json.dumps(value, default=lambda obj: obj.model_dump(mode="json")))


def main():
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == SOURCE_COMMIT
    sys.path.insert(0, str(Path.cwd()))
    import litellm

    import dspy
    from dspy.clients.cache import Cache
    from dspy.utils.usage_tracker import track_usage

    assert Path(dspy.__file__).resolve().is_relative_to(Path.cwd())
    output_dir = Path(sys.argv[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {"name": "text", "init": {"model": "openai/gpt-4o-mini", "temperature": 0.7, "max_tokens": 32},
         "call": {"prompt": "Say hello."}, "texts": ["Hello."]},
        {"name": "rich_n", "init": {"model": "openai/gpt-4o-mini", "temperature": 0.7, "max_tokens": 32},
         "call": {"messages": [{"role": "user", "content": "Check Paris."}], "n": 2, "logprobs": True,
                  "rollout_id": 7}, "texts": ["First answer.", "Second answer."]},
        {"name": "reasoning_key", "init": {"model": "openai/o3-mini", "temperature": 1.0, "max_tokens": 16000},
         "call": {"prompt": "What is two plus two?"}, "texts": ["Four."]},
    ]
    cases = [{**copy.deepcopy(case), "mode": mode} for case in cases for mode in ("sync", "async")]
    for case in cases:
        rich = case["name"] == "rich_n"
        case["name"] += "_" + case["mode"]
        choices = []
        for index, text in enumerate(case.pop("texts")):
            message = {"role": "assistant", "content": text}
            choice = {"index": index, "finish_reason": "stop", "message": message}
            if rich:
                message.update({
                    "reasoning_content": "Check the evidence.",
                    "tool_calls": [{"id": f"call_{index}", "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}],
                    "provider_specific_fields": {"citations": [[{"type": "char_location", "cited_text": "Paris",
                                                                "document_index": 0, "start_char_index": 0,
                                                                "end_char_index": 5}]]},
                })
                choice["logprobs"] = {"content": [{"token": "First", "logprob": -0.1, "bytes": [70],
                                                   "top_logprobs": []}]}
            choices.append(choice)
        case["provider_response"] = {"id": f"fixture-{case['name']}", "created": 1,
                                     "model": case["init"]["model"], "choices": choices,
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
        with tempfile.TemporaryDirectory() as directory:
            cache = Cache(True, True, directory)
            dspy.cache = cache
            response = litellm.ModelResponse(**case["provider_response"])
            response._hidden_params["response_cost"] = 0.01
            lm = dspy.LM(**case["init"])
            backend = importlib.import_module("dspy.clients.lm")._get_litellm()
            method = "completion" if case["mode"] == "sync" else "acompletion"

            def invoke():
                return lm(**case["call"]) if case["mode"] == "sync" else asyncio.run(lm.acall(**case["call"]))

            with patch.object(backend, method, return_value=copy.deepcopy(response)) as completion:
                with track_usage() as usage:
                    case["outputs"] = plain(invoke())
                case["cold_usage"] = usage.get_total_tokens()
                case["cold_history"] = plain({key: lm.history[-1][key] for key in HISTORY_KEYS})
                with track_usage() as usage:
                    assert plain(invoke()) == case["outputs"]
                case["warm_usage"] = usage.get_total_tokens()
                case["warm_history"] = plain({key: lm.history[-1][key] for key in HISTORY_KEYS})
                assert completion.call_count == 1
            assert len(cache.memory_cache) == 1
            case["cache_key"] = next(iter(cache.memory_cache))
            cache.disk_cache.close()
            with zipfile.ZipFile(output_dir / f"{case['name']}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(Path(directory).rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(directory))
    manifest = {"source_commit": SOURCE_COMMIT, "python": platform.python_version(),
                "dependencies": {name: importlib.metadata.version(name)
                                 for name in ("dspy", "litellm", "diskcache", "cloudpickle", "pydantic", "orjson")},
                "history_keys": HISTORY_KEYS, "cases": cases}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
