import dspy

from dspy.primitives.program import handle_async


class Adapter:
    async def _async_call(
        self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True
    ):
        """Internal async implementation"""
        # Format inputs - formatting is sync operation
        inputs = self.format(signature, demos, inputs)
        inputs = (
            dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)
        )

        assert dspy.settings.async_mode, "Async mode is not enabled"
        outputs = await lm.acall(**inputs, **lm_kwargs)

        # Parse outputs - parsing is sync operation
        values = []
        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            assert set(value.keys()) == set(
                signature.output_fields.keys()
            ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)

        return values

    def _sync_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        """Internal async implementation"""
        # Format inputs - formatting is sync operation
        inputs = self.format(signature, demos, inputs)
        inputs = (
            dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)
        )

        # Get completions - this is the async part
        outputs = lm(**inputs, **lm_kwargs)

        # Parse outputs - parsing is sync operation
        values = []
        for output in outputs:
            value = self.parse(signature, output, _parse_values=_parse_values)
            assert set(value.keys()) == set(
                signature.output_fields.keys()
            ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
            values.append(value)

        return values

    @handle_async
    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        """
        Main entry point that handles both sync and async calls.
        Uses handle_async decorator to automatically switch between modes.
        """
        if dspy.settings.async_mode:
            return self._async_call(
                lm, lm_kwargs, signature, demos, inputs, _parse_values
            )
        return self._sync_call(lm, lm_kwargs, signature, demos, inputs, _parse_values)
