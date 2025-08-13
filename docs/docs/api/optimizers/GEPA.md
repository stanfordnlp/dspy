# dspy.GEPA: Reflective Prompt Optimizer

**GEPA** (Genetic-Pareto) is a reflective optimizer proposed in "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457)), that adaptively evolves _textual components_ (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with a text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score.

<!-- START_API_REF -->
::: dspy.GEPA
    handler: python
    options:
        members:
            - compile
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## API Reference: `dspy.GEPA`

`dspy.GEPA` is a fully pluggable, DSPy-native optimizer that can be used to tune **modular programs**, by reflecting on execution traces, leveraging both metric-driven feedback and LLM-powered insight.

### Arguments

- **metric** (*callable*)  
  The feedback and scoring function, invoked as `metric(gold, pred, trace=None, pred_name=None, pred_trace=None)`. This must return either a scalar float (the score) or a dictionary `{score: float, feedback: str}`. You may optionally implement predictor-level feedback for compound systems by handling the `pred_name` and `pred_trace` arguments.

- **reflection_lm** (`dspy.LM` required)  
  A powerful LLM instance used for natural language reflection and propsing new prompts. E.g. `dspy.LM("gpt-5", temperature=1.0, max_tokens=32000)`.

- **Budget configuration** *(exactly one required):*
    - **auto** (*Literal["light", "medium", "heavy"]*): High-level rollout budget; recommended for most users.
    - **max_full_evals** (*int*): Max number of full program evaluations allowed.
    - **max_metric_calls** (*int*): Max raw metric calls permitted (total system executions + per-instance feedback calls).

- **Reflection settings:**
    - **reflection_minibatch_size** (*int*, default=3): Examples per reflection/mutation minibatch.
    - **candidate_selection_strategy** ("pareto" | "current_best", default="pareto"): Strategy for choosing candidates to mutate. Pareto selection is recommended.
    - **skip_perfect_score**: Whether to skip reflection if all minibatch scores are perfect.
    - **add_format_failure_as_feedback**: Whether to treat parse/format failures as actionable feedback.

- **Merge/crossover:**
    - **use_merge** (*bool*, default=True): Enable system-aware merge/crossover between lineages.
    - **max_merge_invocations** (*int*, default=5): Limits on merge step calls.

- **Evaluation config:**
    - **num_threads** (int): Parallelism for metric calls and rollout evaluation.
    - **failure_score** (*float*, default=0.0): Score assigned to failed rollouts.
    - **perfect_score** (*float*, default=1.0): The best achievable metric value (used for early stopping).

- **Logging/config:**
    - **log_dir**: Directory to save run artifacts, checkpoints, logs.
    - **track_stats**: Track and return all candidate programs and detailed stats as `detailed_results` attribute on optimized program.
    - **use_wandb**, **wandb_api_key**, **wandb_init_kwargs**: For enabling Weights & Biases logging.
    - **track_best_outputs** (*bool*): Store best outputs for each instance in validation set.

- **Reproducibility:**
    - **seed** (*int*): Random seed for candidate/rollout selection.

### Usage Example

See GEPA usage tutorials in [GEPA Tutorials](../../tutorials/gepa_ai_program/index.md).

---

## How Does GEPA Work?

### 1. **Reflective Prompt Mutation**

GEPA uses LLMs to _reflect_ on structured execution traces (inputs, outputs, failures, feedback), targeting a chosen module and proposing a new instruction/program text tailored to real observed failures and rich textual/environmental feedback. This moves beyond black-box mutation: each update is grounded in understanding the full context, success/failure reasons, and domain-specific insights.

### 2. **Rich Textual Feedback as Optimization Signal**

GEPA can leverage _any_ textual feedback available—not just scalar rewards. This includes evaluation logs, code traces, failed parses, constraint violations, error message strings, or even isolated submodule-specific feedback. This allows actionable, domain-aware optimization even for compositional, multi-layer systems.

### 3. **Pareto-based Candidate Selection**

Rather than evolving just the _best_ global candidate (which leads to local optima or stagnation), GEPA maintains a Pareto frontier: the set of candidates which achieve the highest score on at least one evaluation instance. In each iteration, the next candidate to mutate is sampled (with probability proportional to coverage) from this frontier, guaranteeing both exploration and robust retention of complementary strategies.

### Algorithm Summary

1. **Initialize** the candidate pool with the seed (usually, the unoptimized system).
2. **Iterate**:
   - **Sample a candidate** (from Pareto frontier).
   - **Sample a minibatch** from the train set.
   - **Select a module** of the candidate for targeted improvement.
   - **Collect execution traces + feedbacks** for module rollout on minibatch.
   - **LLM Reflection:** Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback.
   - **Roll out the new candidate** on the minibatch; **if improved, evaluate on Pareto validation set**.
   - **Update the candidate pool/Pareto frontier.**
   - **[Optionally] System-aware merge/crossover**: Combine best-performing modules from distinct lineages.
3. **Continue** until rollout or metric budget is exhausted. **Return** candidate with best aggregate performance on validation.

---

## Implementing Feedback Metrics

A well-designed metric is central to GEPA's sample efficiency and learning signal richness. It should:

```python
def metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | dict:
    """
    Evaluates the output (pred) relative to the reference/gold, and (optionally) uses
    execution traces for richer feedback. If pred_name/pred_trace are provided, can produce
    predictor/module-specific feedback.
    Returns either a scalar or a dict: {'score': float, 'feedback': str}.
    """
    ...
```
You may simply return `score`, or for richer learning, include actionable text `feedback`. See API code for usage.

---

## Inference-Time Search

GEPA can act as a test-time/inference search mechanism. By setting your `valset` to your _evaluation batch_ and using `track_best_outputs=True`, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search—yielding a "Pareto front" across the batch.

```python
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
pareto_scores = new_prog.detailed_results.val_aggregate_scores
best_outputs = new_prog.detailed_results.best_outputs_valset
```

---

## Example: GEPA's Prompt Meta-Prompt

When mutating a module's instructions, GEPA uses a special reflection prompt. For example:

```
I provided an assistant with the following instructions to perform a task for me:
<current instruction>

The following are examples of task inputs, with outputs and environmental feedback:
<inputs, outputs, feedback>

Your task: Write a new instruction for the assistant. Infer niche/domain-specific knowledge from the feedback, and synthesize explicit strategies and lessons. Place your new instruction in triple backticks.
```

---

## Reference & Citation

If you use the `dspy.GEPA` optimizer, please cite:

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning}, 
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## Further Reading

- [GEPA Paper: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
- [GEPA Github](https://github.com/gepa-ai/gepa)
- [DSPy Documentation](https://dspy.ai/api/optimizers/GEPA/)
- [GEPAAdapter reference](https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa/gepa_utils.py)

For the latest, see [gepa-ai/gepa](https://github.com/gepa-ai/gepa) and [DSPy API reference](https://dspy.ai/api/optimizers/GEPA/).