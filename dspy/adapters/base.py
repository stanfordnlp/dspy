class Adapter:
    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        inputs_ = self.format(signature, demos, inputs)
        inputs_ = dict(prompt=inputs_) if isinstance(inputs_, str) else dict(messages=inputs_)

        outputs = lm(**inputs_, **lm_kwargs)
        values = []

        for output in outputs:
            try:
                value = self.parse(signature, output, _parse_values=_parse_values)
                assert set(value.keys()) == set(signature.output_fields.keys()), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            except Exception as e:
                from .json_adapter import JsonAdapter
                if _parse_values and not isinstance(self, JsonAdapter):
                    return JsonAdapter()(lm, lm_kwargs, signature, demos, inputs, _parse_values=_parse_values)
                else:
                    raise e

            values.append(value)
        
        return values
