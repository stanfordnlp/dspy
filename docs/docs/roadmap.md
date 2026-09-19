# DSPy's direction

DSPy exists to make intelligence **programmable**.

Models will keep improving. That is good news, but a capable model is not yet a system. People build systems so behavior can be repeated, inspected, changed, reused, evaluated, and improved. DSPy provides the programming model for doing that with language models.

This page is both a direction and a filter. We use the principles below to decide what belongs in DSPy's core, what belongs in an integration or application, and what we should research next.

## The design promise

A DSPy program should let you express intent in the form that fits it best:

1. **Python for structure.** Use ordinary code for control flow, state, constraints, and operations that must happen exactly.
2. **Signatures for judgment.** Use typed, named inputs and outputs for transformations where natural-language understanding is useful and some ambiguity is intentional.
3. **Examples and metrics for feedback.** Use data, checks, and judgments to show what success means across real cases, especially the long tail.

You do not need all three on day one. A useful program may start with one signature. Add Python structure when the workflow demands it, and add evaluation or optimization when you can describe what better means.

DSPy keeps these forms separate but composable:

- A **Signature** says what transformation you want.
- A **Module** chooses how to perform it at inference time.
- An **optimizer** uses feedback to improve the program as a whole.

This separation is the core of DSPy. The signature should not need to change when you replace `Predict` with `ReAct`, switch model providers, or adopt a better optimizer.

## Keep structure in Python and judgment in signatures

DSPy is plain Python on purpose. Branches, loops, retries, permissions, and calls to normal software should remain real code. When a step requires fuzzy interpretation or generation, isolate that ambiguity in a well-named signature. In programming-language terms, this is an imperative shell with declarative leaves.

This gives programs deliberate joints: each fuzzy transformation has an interface, each stage can be inspected independently, and the surrounding system remains understandable with normal programming tools. DSPy should complement Python rather than replace it with a new graph language.

Signatures do not discover product intent for you. They are where **you** state it. Put requirements you already know in the signature, types, or surrounding code. Use examples, metrics, and optimizers to handle variation, model-specific behavior, and cases that are difficult to enumerate—not to avoid specifying known requirements.

## Stable programs, improving implementations

DSPy separates a relatively stable, human-facing program from the techniques used to execute and optimize it.

Inference strategies and optimization algorithms should improve—and eventually be replaced—as models improve. A program written against DSPy's core abstractions should benefit without being redesigned around every new prompting trick, model API, or training method.

Although prompt optimization is an important technique, it is not the boundary of DSPy. An optimizer may synthesize instructions, select demonstrations, tune weights, use reinforcement learning, or apply methods that have not been invented yet. The durable contract is: **take a program plus a way to judge it, and produce a better program**.

## What belongs in DSPy

A feature is a strong candidate for DSPy's core when it does one or more of the following while preserving the design promise:

- makes signatures, modules, tools, metrics, or optimizers compose more naturally;
- improves portability across models, providers, and modalities;
- helps isolate and inspect fuzzy LM behavior inside a larger program;
- lets a new inference or learning strategy work across many existing programs;
- makes evaluation, optimization, saving, loading, or deployment reliable end to end;
- removes incidental prompt or provider machinery from user code;
- stays approachable for a Python user who is not an ML specialist.

A feature probably belongs outside the core when it:

- encodes one application's workflow rather than a reusable programming primitive;
- exposes a temporary prompting trick as a permanent user-facing abstraction;
- duplicates Python control flow with a DSPy-specific graph or orchestration language;
- couples task intent to one model, provider, agent architecture, or optimizer;
- hides behavior in a way that makes stages harder to inspect, evaluate, or replace;
- adds convenience at the cost of weakening composition or creating a second way to express the same idea.

Such work can still be valuable. It may fit better as a tool, adapter, integration, recipe, external package, or application built with DSPy.

## Roadmap priorities

We prioritize work that closes the gap between a user's stated intent and the best systems current models can deliver:

1. **Make the small core excellent.** Keep signatures, modules, adapters, tools, metrics, and program state predictable, typed, composable, and easy to debug.
2. **Improve whole-program optimization.** Advance prompt, demonstration, weight, and reinforcement-learning optimizers while reducing their data, compute, and expert-tuning requirements.
3. **Support richer inference without changing intent.** Let programs adopt agents, tools, multimodal models, long-context strategies, and future techniques behind stable signatures.
4. **Complete the path to production.** Make evaluation, observability, async execution, streaming, versioned artifacts, deployment, and monitoring first-class parts of the workflow.
5. **Teach the programming model clearly.** Help software developers and domain experts build strong systems without first becoming prompt engineers or ML researchers.
6. **Grow an open research ecosystem.** Make new modules and optimizers reusable so advances from researchers, model builders, and practitioners can benefit the same programs.

We expect the algorithms under these priorities to change. We intend the programming model above to remain recognizable.
