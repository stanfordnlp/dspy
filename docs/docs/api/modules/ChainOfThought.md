# dspy.ChainOfThought

`ChainOfThought` defaults to `dspy.Reasoning`: it enables native reasoning on supported LMs without requesting a reasoning field in the answer. On other LMs, it requests a short `plan` before the task outputs and returns it as `result.reasoning`. If your signature already has a `plan` field, DSPy uses an unused name such as `plan_2` for this fallback. Native reasoning defaults to `reasoning_effort="low"`; configure `reasoning_effort` on the LM to override it. An explicit `None` or `"none"` selects the plan fallback instead. Models that always reason may still use native reasoning internally.

The public output is still `result.reasoning`. It contains a `dspy.Reasoning` value, or `None` if the provider does not return a native reasoning summary. This is not a guaranteed explanation of the answer. Use `str(result.reasoning)` when present to get text. Existing demonstrations keep their `reasoning` keys.

Pass `rationale_field_type=str` to retain an explicit string-valued reasoning field, or supply a custom `rationale_field` to control its description. These explicit fields do not disable reasoning already configured on the LM. No separate reasoning-cache setting is required.

<!-- START_API_REF -->
::: dspy.ChainOfThought
    handler: python
    options:
        members:
            - __call__
            - acall
            - aforward
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_lm
            - inspect_history
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
