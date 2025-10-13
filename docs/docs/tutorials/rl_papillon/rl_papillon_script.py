"""
Runnable PAPILLON RL script using Arbor + ArborGRPO.

Prereqs:
- pip install -U arbor-ai datasets
- OPENAI_API_KEY must be set for the judge model (openai/gpt-4.1-mini).
"""

import os
import random
import time

import dspy
from datasets import load_dataset
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.clients.utils_finetune import MultiGPUConfig
from dspy.teleprompt.arbor_grpo import ArborGRPO, ArborGRPOConfig

import arbor


# ------------------------
# Fixed run configuration
# ------------------------
SEED = 7

# Models
LOCAL_LM_NAME = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_LM_NAME = "openai/gpt-4.1-mini"  # requires OPENAI_API_KEY

# Dataset sizes for ~5–10 min run on 4 GPUs
TRAIN_SIZE = 96
DEV_SIZE = 48
TEST_SIZE = 48

# GRPO loop settings (speed vs. quality)
NUM_TRAIN_STEPS = 40
NUM_DSPY_EXAMPLES_PER_STEP = 2
NUM_SAMPLES_PER_INPUT = 2  # also used as ArborGRPOConfig.num_generations

# Hardware
NUM_TRAINING_GPUS = 2
NUM_INFERENCE_GPUS = 1

# Finetuning knobs
USE_LORA = True
BF16 = True


random.seed(SEED)


# ------------------------
# PAPILLON module and judges
# ------------------------
class CraftRedactedRequest(dspy.Signature):
    """
    Given a private user query, create a privacy-preserving request for a powerful external LLM.
    The LLM may assist without learning private information about the user.
    """
    user_query = dspy.InputField()
    llm_request = dspy.OutputField()


class RespondToQuery(dspy.Signature):
    """
    Respond to a user query.
    For inspiration, we found a potentially related request to a powerful external LLM and its response.
    """
    related_llm_request = dspy.InputField()
    related_llm_response = dspy.InputField(desc="information from a powerful LLM responding to a related request")
    user_query = dspy.InputField(desc="the user's request you need to fulfill")
    response = dspy.OutputField(desc="your final response to the user's request")


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
                related_llm_request=llm_request, related_llm_response=llm_response, user_query=user_query
            ).response
        except Exception:
            return dspy.Prediction(llm_request="", llm_response="", response="")
        return dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)


class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """
    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that (i) are forms of PII and (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """
    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set((pii_str or "").split("||"))) if pii_str else []
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked if updated_query else 0
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)


def compute_metrics_factory(llm_judge):
    def compute_metrics(gold, pred, trace=None):
        return llm_judge(
            user_query=gold.user_query,
            new_resp=pred.response,
            og_resp=gold.target_response,
            updated_query=pred.llm_request,
            pii_str=gold.pii_str,
        )

    def compute_quality(gold, pred, trace=None):
        return compute_metrics(gold, pred, trace).quality

    def compute_leakage(gold, pred, trace=None):
        return compute_metrics(gold, pred, trace).leakage

    def compute_overall_score(gold, pred, trace=None):
        metrics = compute_metrics(gold, pred, trace)
        overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
        return overall_score >= 1.0 if trace is not None else overall_score

    return compute_overall_score, compute_quality, compute_leakage


# ------------------------
# Data
# ------------------------
def load_pupa_small(train_n=TRAIN_SIZE, dev_n=DEV_SIZE, test_n=TEST_SIZE):
    pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")
    examples = [
        dspy.Example(
            {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
        ).with_inputs("user_query")
        for x in pupa_new["train"]
    ]
    total = train_n + dev_n + test_n
    if len(examples) < total:
        # repeat to meet size (ok for short smoke run)
        m = (total + len(examples) - 1) // len(examples)
        examples = (examples * m)[: total + 1]
    trainset = examples[:train_n]
    devset = examples[train_n : train_n + dev_n]
    testset = examples[train_n + dev_n : train_n + dev_n + test_n]
    print(f"Loaded {len(trainset)} train, {len(devset)} dev, {len(testset)} test.")
    return trainset, devset, testset


# ------------------------
# Main
# ------------------------
def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is required for the judge model.")

    # Start Arbor server and local LM
    arbor_server_info = arbor.init()
    api_base = arbor_server_info["api_base"]

    local_lm = dspy.LM(
        model=f"openai/arbor:{LOCAL_LM_NAME}",
        provider=ArborProvider(),
        temperature=0.7,
        api_base=api_base,
    )
    dspy.configure(lm=local_lm)

    # Judge LM
    openai_lm = dspy.LM(model=JUDGE_LM_NAME)

    # Build program and judges
    papillon = PAPILLON(untrusted_model=openai_lm)
    llm_judge = LLMJudge()
    llm_judge.set_lm(openai_lm)
    compute_overall_score, compute_quality, compute_leakage = compute_metrics_factory(llm_judge)

    # Data
    trainset, devset, testset = load_pupa_small()

    # Zero-shot quick eval
    print("\nZero-shot evaluation (dev)...")
    evaluate = dspy.Evaluate(
        metric=compute_overall_score,
        devset=devset,
        num_threads=min(16, os.cpu_count() or 8),
        display_progress=True,
        display_table=5,
        max_errors=50,
    )
    try:
        evaluate(papillon)
    except Exception as e:
        print(f"Zero-shot evaluation error (continuing): {e}")

    # Switch PAPILLON to local trainable LM
    papillon.set_lm(local_lm)

    # Arbor GRPO config (no env vars)
    grpo_config = ArborGRPOConfig(
        num_generations=NUM_SAMPLES_PER_INPUT,
        temperature=1.0,
        beta=0.03,
        num_iterations=1,
        per_device_train_batch_size=2,
        learning_rate=2e-6,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="constant_with_warmup",
        max_prompt_length=512,
        max_completion_length=192,
        bf16=BF16,
        scale_rewards=True,
        max_grad_norm=0.5,
        lora=USE_LORA,
        report_to="none",
        log_completions=False,
        logging_steps=25,
        max_context_length=None,
    )

    compiler = ArborGRPO(
        metric=compute_overall_score,
        multitask=True,
        num_dspy_examples_per_grpo_step=NUM_DSPY_EXAMPLES_PER_STEP,
        num_samples_per_input=NUM_SAMPLES_PER_INPUT,
        exclude_demos=True,
        num_train_steps=NUM_TRAIN_STEPS,
        num_threads=min(24, os.cpu_count() or 8),
        use_train_as_val=False,
        num_steps_for_val=max(1, NUM_TRAIN_STEPS // 3),
        config=grpo_config,
        gpu_config=MultiGPUConfig(num_inference_gpus=NUM_INFERENCE_GPUS, num_training_gpus=NUM_TRAINING_GPUS),
        report_train_scores=False,
        failure_score=0.0,
    )

    print("\nStarting GRPO optimization...")
    t0 = time.time()
    optimized_papillon = compiler.compile(student=papillon, trainset=trainset, valset=devset)
    print(f"GRPO training took {(time.time() - t0)/60.0:.1f} min")

    print("\nPost-training evaluation (dev)...")
    try:
        evaluate(optimized_papillon)
    except Exception as e:
        print(f"Post-training evaluation error (continuing): {e}")

    if len(devset) > 0:
        ex = devset[0]
        print("\nSample output after training:")
        print(optimized_papillon(**ex.inputs()))


if __name__ == "__main__":
    main()