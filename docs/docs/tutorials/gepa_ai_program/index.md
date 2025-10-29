# Reflective Prompt Evolution with GEPA

This section introduces GEPA, a reflective prompt optimizer for DSPy. GEPA works by leveraging LM's ability to reflect on the DSPy program's trajectory, identifying what went well, what didn't, and what can be improved. Based on this reflection, GEPA proposes new prompts, building a tree of evolved prompt candidates, accumulating improvements as the optimization progresses. Since GEPA can leverage domain-specific text feedback (as opposed to only the scalar metric), GEPA can often propose high performing prompts in very few rollouts. GEPA was introduced in the paper [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) and available as `dspy.GEPA` which internally uses the GEPA implementation provided in [gepa-ai/gepa](https://github.com/gepa-ai/gepa).

## `dspy.GEPA` Tutorials

### [GEPA for AIME (Math)](../gepa_aime/index.ipynb)
This tutorial explores how GEPA can optimize a single `dspy.ChainOfThought` based program to achieve 10% gains on AIME 2025 with GPT-4.1 Mini!

### [GEPA for Structured Information Extraction for Enterprise Tasks](../gepa_facilitysupportanalyzer/index.ipynb)
This tutorial explores how GEPA leverages predictor-level feedback to improve GPT-4.1 Nano's performance on a three-part task for structured information extraction and classification in an enterprise setting.

### [GEPA for Privacy-Conscious Delegation](../gepa_papillon/index.ipynb)
This tutorial explores how GEPA can improve rapidly in as few as 1 iteration, while leveraging a simple feedback provided by a LLM-as-a-judge metric. The tutorial also explores how GEPA benefits from the textual feedback showing a breakdown of aggregate metrics into sub-components, allowing the reflection LM to identify what aspects of the task need improvement.

### [GEPA for Code Backdoor Classification (AI control)](../gepa_trusted_monitor/index.ipynb)
This tutorial explores how GEPA can optimize a GPT-4.1 Nano classifier to identify backdoors in code written by a larger LM, using `dspy.GEPA` and a comparative metric! The comparative metric allows the prompt optimizer to create a prompt that identifies the signals in the code that are indicative of a backdoor, teasing apart positive samples from negative samples.
