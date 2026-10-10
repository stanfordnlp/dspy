"""Regression test: a batch_fn output/input count mismatch must surface as an
exception in the caller, not hang it forever.

https://github.com/stanfordnlp/dspy/issues/10313
"""
import threading

from dspy.utils.unbatchify import Unbatchify


def test_short_batch_output_raises_in_caller():
    def flaky_batch_fn(items):
        # Simulates a batched endpoint that silently drops the last item.
        return items[:-1]

    with Unbatchify(flaky_batch_fn, max_batch_size=8, max_wait_time=0.01) as unbatch:
        result = {}

        def call():
            try:
                unbatch("a")
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(timeout=10.0)

    assert not thread.is_alive(), "caller hung forever on an unresolved Future"
    assert "error" in result
    assert "cannot unbatch" in str(result["error"]) or "outputs for" in str(result["error"])
