import dspy

from dspy.primitives.program import handle_async
from dspy.utils.callback import with_callbacks


class Adapter:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

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
        try:
            values = []
            for output in outputs:
                value = self.parse(signature, output, _parse_values=_parse_values)
                assert set(value.keys()) == set(
                    signature.output_fields.keys()
                ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
                values.append(value)

            return values
        except Exception as e:
            from .json_adapter import JsonAdapter

            if _parse_values and not isinstance(self, JsonAdapter):
                return JsonAdapter()(
                    lm, lm_kwargs, signature, demos, inputs, _parse_values=_parse_values
                )
            raise e

    def _sync_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        """Internal async implementation"""
        # Format inputs - formatting is sync operation
        inputs = self.format(signature, demos, inputs)
        inputs = (
            dict(prompt=inputs) if isinstance(inputs, str) else dict(messages=inputs)
        )

        outputs = lm(**inputs, **lm_kwargs)

        # Parse outputs - parsing is sync operation
        try:
            values = []
            for output in outputs:
                value = self.parse(signature, output, _parse_values=_parse_values)
                assert set(value.keys()) == set(
                    signature.output_fields.keys()
                ), f"Expected {signature.output_fields.keys()} but got {value.keys()}"
                values.append(value)

            return values
        except Exception as e:
            from .json_adapter import JsonAdapter

            if _parse_values and not isinstance(self, JsonAdapter):
                return JsonAdapter()(
                    lm, lm_kwargs, signature, demos, inputs, _parse_values=_parse_values
                )
            raise e

    @handle_async
    def __call__(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
        """
        Main entry point that handles both sync and async calls.
        Uses handle_async decorator to automatically switch between modes.
        """
        if dspy.settings.async_mode:
            return self._async_call(
                lm,
                lm_kwargs,
                signature,
                demos,
                inputs,
                _parse_values,
            )

        return self._sync_call(
            lm,
            lm_kwargs,
            signature,
            demos,
            inputs,
            _parse_values,
        )
