import inspect
import json
import typing
from typing import Annotated, Callable, List, Tuple, Union  # noqa: UP035

import pydantic
import ujson
from pydantic.fields import FieldInfo

import dspy
from dsp.adapters import passages2text
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature, make_signature


def predictor(*args: tuple, **kwargs) -> Callable[..., dspy.Module]:
    def _predictor(func) -> dspy.Module:
        """Decorator that creates a predictor module based on the provided function."""
        signature = _func_to_signature(func)
        *_, output_key = signature.output_fields.keys()
        return _StripOutput(TypedPredictor(signature, **kwargs), output_key)

    # if we have only a single callable argument, the decorator was invoked with no key word arguments
    #  so we just return the wrapped function
    if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
        return _predictor(args[0])
    return _predictor


def cot(*args: tuple, **kwargs) -> Callable[..., dspy.Module]:
    def _cot(func) -> dspy.Module:
        """Decorator that creates a chain of thought module based on the provided function."""
        signature = _func_to_signature(func)
        *_, output_key = signature.output_fields.keys()
        return _StripOutput(TypedChainOfThought(signature, **kwargs), output_key)

    # if we have only a single callable argument, the decorator was invoked with no key word arguments
    #  so we just return the wrapped function
    if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
        return _cot(args[0])
    return _cot


class _StripOutput(dspy.Module):
    def __init__(self, predictor, output_key):
        super().__init__()
        self.predictor = predictor
        self.output_key = output_key

    def copy(self):
        return _StripOutput(self.predictor.copy(), self.output_key)

    def forward(self, **kwargs):
        prediction = self.predictor(**kwargs)
        return prediction[self.output_key]


class FunctionalModule(dspy.Module):
    """To use the @cot and @predictor decorators, your module needs to inheret form this class."""

    def __init__(self):
        super().__init__()
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, dspy.Module):
                self.__dict__[name] = attr.copy()


def TypedChainOfThought(signature, instructions=None, reasoning=None, *, max_retries=3) -> dspy.Module:  # noqa: N802
    """Just like TypedPredictor, but adds a ChainOfThought OutputField."""
    signature = ensure_signature(signature, instructions)
    output_keys = ", ".join(signature.output_fields.keys())

    default_rationale = dspy.OutputField(
        prefix="Reasoning: Let's think step by step in order to",
        desc="${produce the " + output_keys + "}. We ...",
    )
    reasoning = reasoning or default_rationale

    return TypedPredictor(
        signature.prepend(
            "reasoning",
            reasoning,
        ),
        max_retries=max_retries,
    )


class TypedPredictor(dspy.Module):
    def __init__(self, signature, instructions=None, *, max_retries=3, wrap_json=False, explain_errors=False):
        """Like dspy.Predict, but enforces type annotations in the signature.

        Args:
            signature: The signature of the module. Can use type annotations.
            instructions: A description of what the model should do.
            max_retries: The number of times to retry the prediction if the output is invalid.
            wrap_json: If True, json objects in the input will be wrapped in ```json ... ```
            explain_errors: If True, the model will try to explain the errors it encounters.
        """
        super().__init__()
        self.signature = ensure_signature(signature, instructions)
        self.predictor = dspy.Predict(signature)
        self.max_retries = max_retries
        self.wrap_json = wrap_json
        self.explain_errors = explain_errors

    def copy(self) -> "TypedPredictor":
        return TypedPredictor(
            self.signature,
            max_retries=self.max_retries,
            wrap_json=self.wrap_json,
            explain_errors=self.explain_errors,
        )

    def __repr__(self):
        """Return a string representation of the TypedPredictor object."""
        return f"TypedPredictor({self.signature})"

    def _make_example(self, type_) -> str:
        # Note: DSPy will cache this call so we only pay the first time TypedPredictor is called.
        schema = json.dumps(type_.model_json_schema())
        if self.wrap_json:
            schema = "```json\n" + schema + "\n```\n"
        json_object = dspy.Predict(
            make_signature(
                "json_schema -> json_object",
                "Make a very succinct json object that validates with the following schema",
            ),
        )(json_schema=schema).json_object
        # We use the model_validate_json method to make sure the example is valid
        try:
            type_.model_validate_json(_unwrap_json(json_object, type_.model_validate_json))
        except (pydantic.ValidationError, ValueError):
            return ""  # Unable to make an example
        return json_object
        # TODO: Another fun idea is to only (but automatically) do this if the output fails.
        # We could also have a more general "suggest solution" prompt that tries to fix the output
        # More directly.
        # TODO: Instead of using a language model to create the example, we can also just use a
        # library like https://pypi.org/project/polyfactory/ that's made exactly to do this.

    def _format_error(
        self,
        error: Exception,
        task_description: Union[str, FieldInfo],
        model_output: str,
        lm_explain: bool,
    ) -> str:
        if isinstance(error, pydantic.ValidationError):
            errors = []
            for e in error.errors():
                fields = ", ".join(map(str, e["loc"]))
                errors.append(f"{e['msg']}: {fields} (error type: {e['type']})")
            error_text = "; ".join(errors)
        else:
            error_text = repr(error)

        if self.explain_errors and lm_explain:
            if isinstance(task_description, FieldInfo):
                args = task_description.json_schema_extra
                task_description = args["prefix"] + " " + args["desc"]
            return (
                error_text
                + "\n"
                + self._make_explanation(
                    task_description=task_description,
                    model_output=model_output,
                    error=error_text,
                )
            )

        return error_text

    def _make_explanation(self, task_description: str, model_output: str, error: str) -> str:
        class Signature(dspy.Signature):
            """I gave my language model a task, but it failed.

            Figure out what went wrong, and write instructions to help it avoid the error next time.
            """

            task_description: str = dspy.InputField(desc="What I asked the model to do")
            language_model_output: str = dspy.InputField(desc="The output of the model")
            error: str = dspy.InputField(desc="The validation error trigged by the models output")
            explanation: str = dspy.OutputField(desc="Explain what the model did wrong")
            advice: str = dspy.OutputField(
                desc="Instructions for the model to do better next time. A single paragraph.",
            )

        # TODO: We could also try repair the output here. For example, if the output is a float, but the
        # model returned a "float + explanation", the repair could be to remove the explanation.

        return dspy.Predict(Signature)(
            task_description=task_description,
            language_model_output=model_output,
            error=error,
        ).advice

    def _prepare_signature(self) -> dspy.Signature:
        """Add formats and parsers to the signature fields, based on the type annotations of the fields."""
        signature = self.signature
        for name, field in self.signature.fields.items():
            is_output = field.json_schema_extra["__dspy_field_type"] == "output"
            type_ = field.annotation
            if is_output:
                if type_ is bool:

                    def parse(x):
                        x = x.strip().lower()
                        if x not in ("true", "false"):
                            raise ValueError("Respond with true or false")
                        return x == "true"

                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (" (Respond with true or false)" if type_ != str else ""),
                        format=lambda x: x if isinstance(x, str) else str(x),
                        parser=parse,
                    )
                elif type_ in (str, int, float):
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (f" (Respond with a single {type_.__name__} value)" if type_ != str else ""),
                        format=lambda x: x if isinstance(x, str) else str(x),
                        parser=type_,
                    )
                elif False:
                    # TODO: I don't like forcing the model to write "value" in the output.
                    if not (inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel)):
                        type_ = pydantic.create_model("Output", value=(type_, ...), __base__=pydantic.BaseModel)
                        to_json = lambda x, type_=type_: type_(value=x).model_dump_json()[9:-1]  # {"value":"123"}
                        from_json = lambda x, type_=type_: type_.model_validate_json('{"value":' + x + "}").value
                        schema = json.dumps(type_.model_json_schema()["properties"]["value"])
                    else:
                        to_json = lambda x: x.model_dump_json()
                        from_json = lambda x, type_=type_: type_.model_validate_json(x)
                        schema = json.dumps(type_.model_json_schema())
                else:
                    # Anything else we wrap in a pydantic object
                    if not (
                        inspect.isclass(type_)
                        and typing.get_origin(type_) not in (list, tuple)  # To support Python 3.9
                        and issubclass(type_, pydantic.BaseModel)
                    ):
                        type_ = pydantic.create_model("Output", value=(type_, ...), __base__=pydantic.BaseModel)
                        to_json = lambda x, type_=type_: type_(value=x).model_dump_json()
                        from_json = lambda x, type_=type_: type_.model_validate_json(x).value
                        schema = json.dumps(type_.model_json_schema())
                    else:
                        to_json = lambda x: x.model_dump_json()
                        from_json = lambda x, type_=type_: type_.model_validate_json(x)
                        schema = json.dumps(type_.model_json_schema())
                    if self.wrap_json:
                        to_json = lambda x, inner=to_json: "```json\n" + inner(x) + "\n```\n"
                        schema = "```json\n" + schema + "\n```"
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (". Respond with a single JSON object. JSON Schema: " + schema),
                        format=lambda x, to_json=to_json: (x if isinstance(x, str) else to_json(x)),
                        parser=lambda x, from_json=from_json: from_json(_unwrap_json(x, from_json)),
                        type_=type_,
                    )
            else:  # If input field
                is_json = False
                format_ = lambda x: x if isinstance(x, str) else str(x)
                if type_ in (List[str], list[str], Tuple[str], tuple[str]):
                    format_ = passages2text
                # Special formatting for lists of known types. Maybe the output fields sohuld have this too?
                elif typing.get_origin(type_) in (List, list, Tuple, tuple):
                    (inner_type,) = typing.get_args(type_)
                    if inspect.isclass(inner_type) and issubclass(inner_type, pydantic.BaseModel):
                        format_ = (
                            lambda x: x if isinstance(x, str) else "[" + ",".join(i.model_dump_json() for i in x) + "]"
                        )
                    else:
                        format_ = lambda x: x if isinstance(x, str) else json.dumps(x)
                    is_json = True
                elif inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel):
                    format_ = lambda x: x if isinstance(x, str) else x.model_dump_json()
                    is_json = True
                if self.wrap_json and is_json:
                    format_ = lambda x, inner=format_: x if isinstance(x, str) else "```json\n" + inner(x) + "\n```\n"
                signature = signature.with_updated_fields(name, format=format_)

        return signature

    def forward(self, **kwargs) -> dspy.Prediction:
        modified_kwargs = kwargs.copy()
        # We have to re-prepare the signature on every forward call, because the base
        # signature might have been modified by an optimizer or something like that.
        signature = self._prepare_signature()
        for try_i in range(self.max_retries):
            result = self.predictor(**modified_kwargs, new_signature=signature)
            errors = {}
            parsed_results = []
            # Parse the outputs
            for completion in result.completions:
                parsed = {}
                for name, field in signature.output_fields.items():
                    try:
                        value = completion[name]
                        parser = field.json_schema_extra.get("parser", lambda x: x)
                        parsed[name] = parser(value)
                    except (pydantic.ValidationError, ValueError) as e:
                        errors[name] = self._format_error(
                            e,
                            signature.fields[name],
                            value,
                            lm_explain=try_i + 1 < self.max_retries,
                        )

                        # If we can, we add an example to the error message
                        current_desc = field.json_schema_extra.get("desc", "")
                        i = current_desc.find("JSON Schema: ")
                        if i == -1:
                            continue  # Only add examples to JSON objects
                        suffix, current_desc = current_desc[i:], current_desc[:i]
                        prefix = "You MUST use this format: "
                        if (
                            try_i + 1 < self.max_retries
                            and prefix not in current_desc
                            and (example := self._make_example(field.annotation))
                        ):
                            signature = signature.with_updated_fields(
                                name,
                                desc=current_desc + "\n" + prefix + example + "\n" + suffix,
                            )
                # No reason trying to parse the general signature, or run more completions, if we already have errors
                if errors:
                    break
                # Instantiate the actual signature with the parsed values.
                # This allow pydantic to validate the fields defined in the signature.
                try:
                    _ = self.signature(**kwargs, **parsed)
                    parsed_results.append(parsed)
                except pydantic.ValidationError as e:
                    errors["general"] = self._format_error(
                        e,
                        signature.instructions,
                        "\n\n".join(
                            "> " + field.json_schema_extra["prefix"] + " " + completion[name]
                            for name, field in signature.output_fields.items()
                        ),
                        lm_explain=try_i + 1 < self.max_retries,
                    )
            if errors:
                # Add new fields for each error
                for name, error in errors.items():
                    modified_kwargs[f"error_{name}_{try_i}"] = error
                    if name == "general":
                        error_prefix = "General:"
                    else:
                        error_prefix = signature.output_fields[name].json_schema_extra["prefix"]
                    number = "" if try_i == 0 else f" ({try_i+1})"
                    signature = signature.append(
                        f"error_{name}_{try_i}",
                        dspy.InputField(
                            prefix=f"Past Error{number} in {error_prefix}",
                            desc="An error to avoid in the future",
                        ),
                    )
            else:
                # If there are no errors, we return the parsed results
                return Prediction.from_completions(
                    {key: [r[key] for r in parsed_results] for key in signature.output_fields},
                )
        raise ValueError(
            "Too many retries trying to get the correct output format. " + "Try simplifying the requirements.",
            errors,
        )


def _func_to_signature(func):
    """Make a dspy.Signature based on a function definition."""
    sig = inspect.signature(func)
    annotations = typing.get_type_hints(func, include_extras=True)
    output_key = func.__name__
    instructions = func.__doc__
    fields = {}

    # Input fields
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        # We default to str as the type of the input
        annotation = annotations.get(param.name, str)
        kwargs = {}
        if typing.get_origin(annotation) is Annotated:
            desc = next((arg for arg in typing.get_args(annotation) if isinstance(arg, str)), None)
            if desc is not None:
                kwargs["desc"] = desc
        fields[param.name] = (annotation, dspy.InputField(**kwargs))

    # Output field
    kwargs = {}
    annotation = annotations.get("return", str)
    if typing.get_origin(annotation) is Annotated:
        desc = next((arg for arg in typing.get_args(annotation) if isinstance(arg, str)), None)
        if desc is not None:
            kwargs["desc"] = desc
    fields[output_key] = (annotation, dspy.OutputField(**kwargs))

    return dspy.Signature(fields, instructions)


def _unwrap_json(output, from_json: Callable[[str], Union[pydantic.BaseModel, str]]):
    try:
        return from_json(output).model_dump_json()
    except (ValueError, pydantic.ValidationError, AttributeError):
        output = output.strip()
        if output.startswith("```"):
            if not output.startswith("```json"):
                raise ValueError("json output should start with ```json") from None
            if not output.endswith("```"):
                raise ValueError("Don't write anything after the final json ```") from None
            output = output[7:-3].strip()
        if not output.startswith("{") or not output.endswith("}"):
            raise ValueError("json output should start and end with { and }") from None
        return ujson.dumps(ujson.loads(output))  # ujson is a bit more robust than the standard json
