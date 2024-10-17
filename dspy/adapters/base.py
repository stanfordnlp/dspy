from dspy.utils.callback import with_callbacks


class Adapter:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        from dspy.utils.callback import with_callbacks

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs = self.format(signature, demos, inputs)
        inputs = dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)

        outputs = lm(**inputs, **lm_kwargs)
        values = []

        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)
        
        return values
