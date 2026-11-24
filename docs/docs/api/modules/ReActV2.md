# dspy.ReActV2

`ReActV2` is a deprecated compatibility subclass of [`dspy.ReAct`](ReAct.md) in DSPy 3.5. It warns on construction and will be removed in DSPy 3.6. Replace `dspy.ReActV2(...)` with `dspy.ReAct(...)`; the constructor and behavior are the same.

The canonical implementation stores calls and results in structured `dspy.History`, supports async execution and multiple sequential tool calls per turn, and submits typed outputs through an internal `submit` tool. Forced submission failure raises rather than returning incomplete outputs. See [ReAct](../../diving-deeper/react.md) for configuration and [migration to DSPy 3.5](../../diving-deeper/react.md#migration-to-dspy-35) for changes from the old trajectory-based implementation, including recompiling old optimized state.

<!-- START_API_REF -->
::: dspy.ReActV2
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
