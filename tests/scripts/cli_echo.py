"""Test CLI script for dspy.CLI tests.

Modes (set via CLI_MODE env var):
- plain (default): Echo stdin to stdout
- json: Emit Codex-style JSONL with agent_message event
- multi_json: Emit multiple JSONL events (thinking + agent_message)
- json_fields: Emit JSON dict with named fields
- fail: Exit with code 2
- timeout: Sleep forever (for timeout tests)
- warn: Print warning to stderr, echo to stdout
- cost: Emit JSONL with cost info
- slow: Sleep CLI_DELAY seconds then echo
"""

import json
import os
import sys
import time

MODE = os.environ.get("CLI_MODE", "plain")


def main():
    payload = sys.stdin.read()
    if not payload:
        print("failed to read input", file=sys.stderr)
        return 3

    content = payload.strip()

    if MODE == "fail":
        print("intentional failure", file=sys.stderr)
        return 2

    if MODE == "timeout":
        time.sleep(3600)  # Sleep forever — test should kill us
        return 0

    if MODE == "warn":
        print("cli warning: proceed with caution", file=sys.stderr)
        sys.stdout.write(content)
        return 0

    if MODE == "slow":
        delay = float(os.environ.get("CLI_DELAY", "0.5"))
        time.sleep(delay)
        sys.stdout.write(content)
        return 0

    if MODE == "json":
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": content},
        }
        sys.stdout.write(json.dumps(event) + "\n")
        return 0

    if MODE == "multi_json":
        events = [
            {"type": "thinking", "content": "Let me think about this..."},
            {"type": "tool_call", "content": "search(query)"},
            {"type": "tool_result", "content": "Found 3 results"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": content}},
        ]
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
        return 0

    if MODE == "json_fields":
        # Parse content as "field1=val1,field2=val2"
        fields = {}
        for pair in content.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                fields[k.strip()] = v.strip()
        sys.stdout.write(json.dumps(fields))
        return 0

    if MODE == "cost":
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": content}},
            {"type": "usage", "cost": 0.05, "tokens": {"input": 100, "output": 50}},
        ]
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
        return 0

    # Default: plain echo
    sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
