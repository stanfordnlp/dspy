# dspy.TypeSafeAdapter

Translates a signature into the single-state, native judgment request accepted
by `dspy.LM("typesafe:jev-1.14.0", engine="lm15")`. Select it explicitly with
`dspy.context(adapter=dspy.TypeSafeAdapter())` or `dspy.configure`.

Instructions, named inputs, and demonstrations form the state. Output field
descriptions become question instructions. Image inputs remain image parts;
the bundled lm15 provider owns transport, the experimental image opt-in,
native question encoding, and provider evidence.

See [BoundingBox](../primitives/BoundingBox.md) for a complete example, supported
output types, limitations, and reproducible live checks. This adapter uses the
bundled native client, not `dspy.experimental.TypeSafe` or `typesafe-sdk`.

<!-- START_API_REF -->
::: dspy.TypeSafeAdapter
    handler: python
    options:
        members:
            - __call__
            - acall
            - format
            - parse
        show_root_heading: true
        heading_level: 2
        show_source: true
        inherited_members: true
<!-- END_API_REF -->
