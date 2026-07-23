import sys
import types
from unittest.mock import MagicMock

from dspy.clients.togetherai import TogetherProvider, TrainingJobTogether


def test_is_provider_model():
    # A real Together model → True.
    assert TogetherProvider.is_provider_model("together_ai/meta-llama/Llama-3-8b") is True
    # A non-Together model → False.
    assert TogetherProvider.is_provider_model("openai/gpt-4o") is False
    # Contains "together_ai" but not at the start → False (guards against startswith -> contains).
    assert TogetherProvider.is_provider_model("openai/my-together_ai-clone") is False


def test_status_returns_none_before_job_starts():
    # No job started yet (job_id is None) → status() returns None instead of hitting the API.
    assert TrainingJobTogether().status() is None


def test_finetune_returns_prefixed_model_name(monkeypatch):
    # 1. Build a fake Together client that answers only the calls finetune() makes.
    fake_client = MagicMock()
    fake_client.files.upload.return_value.id = "file-abc"  # Step 2: upload → file id
    fake_client.fine_tuning.create.return_value.id = "job-xyz"  # Step 3: create → job id
    fake_client.fine_tuning.retrieve.return_value.status = "completed"  # Step 4: poll → "completed"
    fake_client.fine_tuning.retrieve.return_value.x_model_output_name = "my-model"  # Step 5: retrieve → name

    # 2. Patch `together` so finetune()'s `from together import Together` grabs our fake.
    fake_together = types.ModuleType("together")
    fake_together.Together = lambda *a, **k: fake_client
    monkeypatch.setitem(sys.modules, "together", fake_together)

    # 3. Call the real finetune() — it runs its full logic against the fake.
    job = TrainingJobTogether()
    result = TogetherProvider.finetune(
        job=job,
        model="together_ai/meta-llama/Llama-3-8b",
        train_data=[{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}],
    )

    # 4. It should return the fine-tuned model's name, DSPy-prefixed.
    assert result == "together_ai/my-model"

    # 5. What went OUT: create() should get the BARE model name, "together_ai/" prefix stripped.
    assert fake_client.fine_tuning.create.call_args.kwargs["model"] == "meta-llama/Llama-3-8b"
