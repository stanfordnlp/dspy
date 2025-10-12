#!/usr/bin/env python3
"""
Quick PAPILLON + GRPO Demo Script
A simplified version of the RL PAPILLON tutorial for fast local GPU testing.
"""

import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.teleprompt.arbor_grpo import ArborGRPO
from dspy.teleprompt.arbor_grpo.arbor_grpo_config import ArborGRPOConfig
from dspy.clients.utils_finetune import MultiGPUConfig
from datasets import load_dataset

print("=" * 80)
print("PAPILLON Quick Demo - Optimized for Fast Local GPU Testing")
print("=" * 80)

# ============================================================================
# 1. Initialize Arbor Server
# ============================================================================
print("\n[1/7] Initializing Arbor server...")
import arbor
arbor_server_info = arbor.init()
print(f"✓ Arbor server initialized at {arbor_server_info['api_base']}")

# ============================================================================
# 2. Configure Language Models
# ============================================================================
print("\n[2/7] Configuring language models...")

# Use a smaller model for faster training
local_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for speed
local_lm = dspy.LM(
    model=f"openai/arbor:{local_lm_name}",
    provider=ArborProvider(),
    temperature=0.7,
    api_base=arbor_server_info["api_base"],
)

dspy.configure(lm=local_lm)

# Use a smaller OpenAI model for cost efficiency
openai_lm = dspy.LM(model="openai/gpt-4o-mini")
print(f"✓ Local model: {local_lm_name}")
print(f"✓ Remote model: gpt-4o-mini")

# ============================================================================
# 3. Define PAPILLON System
# ============================================================================
print("\n[3/7] Setting up PAPILLON system...")


class CraftRedactedRequest(dspy.Signature):
    """Create a privacy-preserving request for a powerful external LLM."""
    user_query = dspy.InputField()
    llm_request = dspy.OutputField()


class RespondToQuery(dspy.Signature):
    """Respond to a user query using information from an external LLM."""
    related_llm_request = dspy.InputField()
    related_llm_response = dspy.InputField(desc="information from external LLM")
    user_query = dspy.InputField(desc="the user's request")
    response = dspy.OutputField(desc="final response to user")


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
        self.respond_to_query = dspy.Predict(RespondToQuery)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        try:
            llm_request = self.craft_redacted_request(user_query=user_query).llm_request
            llm_response = self.untrusted_model(llm_request)[0]
            response = self.respond_to_query(
                related_llm_request=llm_request,
                related_llm_response=llm_response,
                user_query=user_query
            ).response
        except Exception as e:
            print(f"Warning: Error in forward pass: {e}")
            return dspy.Prediction(llm_request="", llm_response="", response="")

        return dspy.Prediction(
            llm_request=llm_request,
            llm_response=llm_response,
            response=response
        )


print("✓ PAPILLON system defined")

# ============================================================================
# 4. Load and Prepare Dataset (REDUCED SIZE)
# ============================================================================
print("\n[4/7] Loading dataset (small subset for quick testing)...")

pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")

# Use much smaller dataset for quick testing
examples = [
    dspy.Example({
        "target_response": x["target_response"],
        "user_query": x["user_query"],
        "pii_str": x["pii_units"]
    }).with_inputs("user_query")
    for x in pupa_new["train"][:50]  # Only use 50 examples total
]

# Split: 30 train, 10 dev, 10 test
trainset = examples[:30]
devset = examples[30:40]
testset = examples[40:50]

print(f"✓ Loaded {len(trainset)} train, {len(devset)} dev, {len(testset)} test examples")

# ============================================================================
# 5. Define Simple Metric (SIMPLIFIED FOR SPEED)
# ============================================================================
print("\n[5/7] Setting up evaluation metric...")


class JudgeQuality(dspy.Signature):
    """Is response A at least as good as response B?"""
    user_query = dspy.InputField()
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


def simple_metric(gold, pred, trace=None):
    """Simplified metric that only checks quality (skips leakage for speed)."""
    try:
        judge = dspy.ChainOfThought(JudgeQuality)
        with dspy.context(lm=openai_lm):
            result = judge(
                user_query=gold.user_query,
                response_A=pred.response,
                response_B=gold.target_response
            ).judgment
        return float(result) if trace is None else result
    except Exception as e:
        print(f"Warning: Metric evaluation failed: {e}")
        return 0.5 if trace is None else False


print("✓ Metric configured (quality-based)")

# ============================================================================
# 6. Optimize with ArborGRPO (FAST SETTINGS)
# ============================================================================
print("\n[6/7] Running GRPO optimization (fast settings)...")
print("This will take approximately 10-15 minutes on a single GPU...")

papillon = PAPILLON(untrusted_model=openai_lm)
papillon.set_lm(local_lm)

# Create ArborGRPO config with fast training settings
grpo_config = ArborGRPOConfig(
    num_generations=4,  # Reduced from 8
    temperature=1.0,
    beta=0.04,
    num_iterations=1,
    per_device_train_batch_size=2,  # Reduced from 8
    learning_rate=2e-6,
    gradient_accumulation_steps=2,  # Reduced from 4
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    lr_scheduler_type="constant_with_warmup",
    max_prompt_length=None,
    max_completion_length=None,
    scale_rewards=True,
    max_grad_norm=0.5,
    lora=True,  # Use LoRA for memory efficiency
    report_to="none",
    log_completions=True,
    logging_steps=1,
)

# Single GPU configuration for local testing
gpu_config = MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1)

compiler = ArborGRPO(
    metric=simple_metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=2,  # Reduced from 4
    num_rollouts_per_grpo_step=4,  # Reduced from 8
    exclude_demos=True,
    num_train_steps=5,  # VERY reduced from 500 for quick testing
    config=grpo_config,
    gpu_config=gpu_config,
)

print("\nStarting training...")
print("Training configuration:")
print(f"  - Training steps: 5 (reduced for speed)")
print(f"  - Examples per step: 2")
print(f"  - Generations per example: 4")
print(f"  - LoRA enabled: True")
print(f"  - Single GPU mode")
print()

optimized_papillon = compiler.compile(
    student=papillon,
    trainset=trainset,
    valset=devset,
)

print("\n✓ Training complete!")

# ============================================================================
# 7. Test the Optimized Model
# ============================================================================
print("\n[7/7] Testing optimized model...")

test_example = devset[0]
print(f"\nTest Query: {test_example.user_query[:100]}...")

result = optimized_papillon(**test_example.inputs())

print(f"\nRedacted Request to External LLM:")
print(f"  {result.llm_request[:200]}...")
print(f"\nFinal Response:")
print(f"  {result.response[:200]}...")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)
print("\nThis was a quick demo with reduced parameters for fast testing.")
print("For full training, increase these parameters in the script:")
print("  - num_train_steps: 5 → 500")
print("  - trainset size: 30 → 225")
print("  - per_device_train_batch_size: 2 → 8")
print("  - num_dspy_examples_per_grpo_step: 2 → 4")
print("  - GPU config: single → multi-GPU")
print("\nExpected training time with full settings: 2-3 hours on 4xH100")
print("=" * 80)

