class Adapter:
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        messages = self.format(signature, demos, inputs)
        outputs = lm(messages=messages, **lm_kwargs)
        values = []

        for output in outputs:
            value = self.parse(signature, output)
            assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)
        
        return values