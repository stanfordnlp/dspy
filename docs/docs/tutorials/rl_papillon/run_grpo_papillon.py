"""
Runnable script version of the rl_papillon/index.ipynb tutorial, updated for
Arbor GRPO Async Rollouts and compatible with ArborTrainConfig-backed kwargs.

This configuration targets a ~5–10 minute run on a single GPU by limiting
dataset size and GRPO steps. Adjust num_train_steps, num_dspy_examples_per_grpo_step,
and num_rollouts_per_grpo_step to tune runtime.
"""

from __future__ import annotations

import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.teleprompt.grpo import GRPO
from dspy.teleprompt.arbor_grpo import ArborTrainConfig


def main():
    # Initialize Arbor server and local LM
    import arbor

    arbor_server_info = arbor.init()  # Background server

    # Robustly determine api_base from different possible return types
    def _extract_api_base(info):
        import os

        # Direct string
        if isinstance(info, str):
            return info
        # Dict-like
        if isinstance(info, dict):
            for key in ("api_base", "base_url", "url"):
                if key in info and isinstance(info[key], str):
                    return info[key]
        # Object with attribute
        for attr in ("api_base", "base_url", "url"):
            if hasattr(info, attr):
                val = getattr(info, attr)
                if isinstance(val, str):
                    return val
        # Tuple/List where first string could be base
        if isinstance(info, (tuple, list)) and info and isinstance(info[0], str):
            return info[0]
        # Environment fallbacks
        env = os.getenv("ARBOR_API_BASE") or os.getenv("ARBOR_BASE_URL")
        if env:
            return env
        raise ValueError(
            "Could not determine Arbor api_base from arbor.init(). Please set ARBOR_API_BASE or pass an explicit base."
        )

    api_base = _extract_api_base(arbor_server_info)

    # Prefer a smaller model to reduce memory usage
    local_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    local_lm = dspy.LM(
        model=f"openai/arbor:{local_lm_name}",
        provider=ArborProvider(),
        temperature=0.7,
        api_base=api_base,
    )
    dspy.configure(lm=local_lm)

    # Optional: external judge LM
    openai_lm = dspy.LM(model="openai/gpt-4.1-mini")

    # Define the program (PAPILLON) and judges (same structure as notebook)
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
        related_llm_response = dspy.InputField(
            desc="information from a powerful LLM responding to a related request"
        )
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
                    related_llm_request=llm_request,
                    related_llm_response=llm_response,
                    user_query=user_query,
                ).response
            except Exception:
                return dspy.Prediction(llm_request="", llm_response="", response="")

            return dspy.Prediction(
                llm_request=llm_request, llm_response=llm_response, response=response
            )

    # Judges/metrics
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
        Count the number information pieces in `pii` that (i) are forms of PII and (ii) are leaked into the prompt.
        Don't count cases where the leaked info is just a celebrity name or where `pii` info doesn't appear.
        """

        pii = dspy.InputField()
        prompt = dspy.InputField()
        num_pii_leaked: int = dspy.OutputField()

    class LLMJudge(dspy.Module):
        def __init__(self):
            self.quality_judge = dspy.ChainOfThought(JudgeQuality)
            self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

        def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
            judgment_1 = self.quality_judge(
                user_query=user_query, response_A=new_resp, response_B=og_resp
            ).judgment
            judgment_2 = self.quality_judge(
                user_query=user_query, response_A=og_resp, response_B=new_resp
            ).judgment
            judgment = judgment_1 or (judgment_1 == judgment_2)

            pii = list(set(pii_str.split("||")))
            pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
            pii_score = pii_score / len(pii) if len(pii) > 0 else 0

            return dspy.Prediction(quality=judgment, leakage=pii_score)

    llm_judge = LLMJudge()
    llm_judge.set_lm(openai_lm)

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

    def compute_overall_score(gold, pred, trace=None):
        metrics = compute_metrics(gold, pred, trace)
        overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
        return overall_score >= 1.0 if trace is not None else overall_score

    # Load a modest subset to fit into 5–10 minutes
    from datasets import load_dataset

    def load_pupa_small(train_n=128, dev_n=32, test_n=0):
        ds = load_dataset("Columbia-NLP/PUPA", "pupa_new")
        train_ds = ds["train"]
        # Select deterministically without materializing dict-of-lists
        total = train_n + dev_n + test_n
        take = min(total, len(train_ds))
        sel = train_ds.select(range(take))
        rows = [sel[i] for i in range(len(sel))]
        examples = [
            dspy.Example(
                {
                    "target_response": r["target_response"],
                    "user_query": r["user_query"],
                    "pii_str": r["pii_units"],
                }
            ).with_inputs("user_query")
            for r in rows
        ]
        # If dataset is smaller than requested, repeat to meet size for quick runs
        if len(examples) < total:
            m = (total + len(examples) - 1) // len(examples)
            examples = (examples * m)[: total]
        trainset = examples[:train_n]
        devset = examples[train_n : train_n + dev_n]
        testset = examples[train_n + dev_n : train_n + dev_n + test_n]
        print(f"Loaded {len(trainset)} train, {len(devset)} dev, {len(testset)} test.")
        return trainset, devset, testset

    trainset, devset, _ = load_pupa_small()

    # Zero-shot evaluation (optional)
    zeroshot = PAPILLON(untrusted_model=openai_lm)
    kwargs = dict(num_threads=8, display_progress=True, display_table=3, max_errors=50)
    evaluate = dspy.Evaluate(metric=compute_overall_score, devset=devset[:16], **kwargs)
    _ = evaluate(zeroshot)

    # Prepare student program
    papillon = PAPILLON(untrusted_model=openai_lm)
    papillon.set_lm(local_lm)

    # Arbor-compatible training config; GRPO will inject num_generations
    arbor_cfg = ArborTrainConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        temperature=1.0,
        beta=0.04,
        learning_rate=2e-6,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_prompt_length=256,
        max_completion_length=256,
        scale_rewards=True,
        max_grad_norm=0.5,
        lora=True,
        log_completions=True,
        logging_steps=10,
        max_steps=200,  # Trainer cap; outer loop controlled by num_train_steps
        generation_batch_size=2,
    )

    # Ad-hoc extra keys accepted by Arbor backend to reduce memory
    train_kwargs = arbor_cfg.to_dict()
    train_kwargs["max_seq_len"] = 1024
    train_kwargs["vllm_gpu_memory_utilization"] = 0.2

    # Configure GRPO with smaller steps/rollouts for ~5–10 minutes
    compiler = GRPO(
        metric=compute_overall_score,
        multitask=True,
        num_dspy_examples_per_grpo_step=1,
        num_rollouts_per_grpo_step=2,
        exclude_demos=True,
        num_train_steps=40,
        num_threads=8,
        use_train_as_val=False,
        num_steps_for_val=10,
        train_kwargs=arbor_cfg.to_dict(),
        report_train_scores=False,
    )

    optimized_papillon = compiler.compile(
        student=papillon,
        trainset=trainset[:64],  # further reduce runtime
        valset=devset[:16],
    )

    # Quick sanity invocation
    example = devset[0]
    print(optimized_papillon(**example.inputs()))


if __name__ == "__main__":
    main()


