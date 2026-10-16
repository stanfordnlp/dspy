# dspy.ReActV2

`ReActV2` is the experimental, native-tool-aware replacement for the current `dspy.ReAct` implementation. It will become `dspy.ReAct` in DSPy 3.5. The `dspy.ReActV2` name will remain as a deprecated compatibility alias throughout the 3.5 release line and will be removed in DSPy 3.6. It stores calls and results in structured `dspy.History`, supports multiple tool calls in one turn, and submits the signature's typed output fields through an internal tool.

For benefits, configuration, differences from ReAct, and the plan to merge the implementation back into the canonical `dspy.ReAct` API, see [ReAct and ReActV2](../../diving-deeper/react.md).

<!-- START_API_REF -->
::: dspy.ReActV2
    handler: python
    options:
        members:
            - __call__
            - acall
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
