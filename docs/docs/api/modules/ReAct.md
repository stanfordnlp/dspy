# dspy.ReAct

In DSPy 3.5, `ReAct(signature, tools, max_iters=20)` runs a structured `dspy.History` loop with typed `submit` outputs, async support through `acall`, and multiple sequential tool calls per model turn. Predictions contain the declared outputs, `history`, and `termination_reason`. If a final forced submission fails, ReAct raises rather than returning incomplete outputs.

The legacy trajectory, extractor, and automatic truncation APIs are removed. Old optimized state must be recompiled for the new predictor schema. See [ReAct](../../diving-deeper/react.md) for configuration and [migration to DSPy 3.5](../../diving-deeper/react.md#migration-to-dspy-35).

<!-- START_API_REF -->
::: dspy.ReAct
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
