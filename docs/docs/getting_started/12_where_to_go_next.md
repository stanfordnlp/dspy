# Where to go next

## Compose harder pipelines

If you want multi-step modules with branching control flow, the \[Modules: composing your own\] guide picks up where Section 7 left off.

## Building richer metrics

Our haiku metric was intentionally simple. Programs usually need composite scores that blend syntax checks, semantic similarity, and/or LLM-as-judge rubrics. The \[Metrics: designing and composing\] guide walks through weighting sub-scores, preventing keyword-stuffing, and validating that your metric truly captures what you care about before you let an optimizer chase it.

## Try a different optimizer

If GEPA didn't fit your task, the \[Optimizers: choosing one\] guide walks through when to reach for `BootstrapFewShot`, `MIPROv2`, `BootstrapFinetune`, and the rest.

## Debug a run

Calls and traces are inspectable with `dspy.inspect_history()` and callbacks — see the \[Observability and debugging\] guide.

## Serve in production

Programs can be made async, streamed, and parallelized. The \[Async, streaming, and parallel\] guide covers the surface.  
