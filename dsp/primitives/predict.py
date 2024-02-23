from collections import Counter
from contextlib import contextmanager
from typing import Callable, Any, Optional

import dsp
from dsp.utils import zipstar, normalize_text
from dsp.primitives.inspect import FuncInspector
from dsp.utils.utils import dotdict
from dsp.templates.template_v3 import Template
from dsp.primitives.demonstrate import Example


class Completions:
    """A state object that holds the valid LM completions for a given Template."""

    def __init__(self, completions: list[Example], template: Template):
        self.data = completions
        self.template = template

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def unpack(self, key=None):
        if key:
            return [getattr(c, key) for c in self.data]

        return zipstar(self.data)

    def __getattr__(self, name):
        assert len(self.data) == 1

        completion = self.data[0]

        if name in completion.keys():
            return getattr(completion, name)

        if name.endswith("s") and name[:-1] in completion.keys():
            pass

        assert False, name


def generate(template: Template, **kwargs) -> Callable:
    """Returns a callable function that generates completions for a given example using the provided template."""
    if hasattr(dsp.settings, "inspect"):
        inspector = dsp.settings.inspect
        _generate = inspector.inspect_func(dsp.predict._generate)
        return _generate(template, **kwargs)
    else:
        return dsp.predict._generate(template, **kwargs)


def _generate(template: Template, **kwargs) -> Callable:
    """Returns a callable function that generates completions for a given example using the provided template."""
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")

    generator = dsp.settings.lm

    def do_generate(
        example: Example, stage: str, max_depth: int = 2, original_example=None
    ):
        if not dsp.settings.lm:
            raise AssertionError("No LM is loaded.")
        original_example = original_example or example
        assert stage is not None

        # Look up the appropriate fields in each demonstration.
        example = example.demos_at(lambda d: d[stage])

        # Generate and extract the fields.
        prompt = template(example)

        completions: list[dict[str, Any]] = generator(prompt, **kwargs)
        completions: list[Example] = [template.extract(example, p) for p in completions]

        # Find the completions that are most complete.
        field_names: list[str] = [field.input_variable for field in template.fields]

        last_field_idx = 0
        for field_idx, key in enumerate(field_names):
            completions_ = [
                c for c in completions if key in c.keys() and c[key] is not None
            ]

            # Filter out completions that are missing fields that are present in at least one completion.
            if len(completions_):
                completions = completions_
                last_field_idx = field_idx + 1

        # If none of the completions is completed (i.e., none has the final field set).
        if last_field_idx < len(field_names):
            # Pick the first completion that has gone farthest.
            completion = completions[0]
            completion[field_names[last_field_idx]] = ""

            # Recurse with greedy decoding and a shorter length.
            max_tokens = kwargs.get("max_tokens", dsp.settings.lm.kwargs["max_tokens"])
            max_tokens = min(max(75, max_tokens // 2), max_tokens)
            new_kwargs = {
                **kwargs,
                "max_tokens": max_tokens,
                "n": 1,
                "temperature": 0.0,
            }

            assert max_depth > 0
            return generate(template, **new_kwargs)(
                completion,
                stage=stage,
                max_depth=max_depth - 1,
                original_example=original_example,
            )

        completions = Completions(completions, template=template)
        example = example.copy(completions=completions)

        if len(completions) == 1:
            completion = completions[0]
            example[stage] = example.copy(**completion)

            if dsp.settings.compiling:
                inputs_ = set(original_example.keys())
                inputs = [
                    f.input_variable
                    for f in template.fields
                    if f.input_variable in inputs_
                ]
                outputs = [
                    f.output_variable
                    for f in template.fields
                    if f.input_variable not in inputs_
                ]

                example.compiling_stages = example.get("compiling_stages", [])
                example.compiling_stages.append(
                    {
                        "name": stage,
                        "template": template,
                        "inputs": inputs,
                        "outputs": outputs,
                    }
                )
        else:
            # assert not dsp.settings.compiling, "TODO: At this point, cannot compile n>1 generations"
            example[stage] = dotdict(completions=completions)

        return example, completions

    return do_generate


def generate_sc(
    example, prompt, normalize=True, extract=None, prediction_field=None, **kwargs
):
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")
    kwargs = {"temperature": 0.7, "n": 20, "max_tokens": 150, **kwargs}

    completions = dsp.settings.lm(prompt, **kwargs)
    completions = extract_final_answer(example, completions, extract=extract)
    return majority_vote_(
        completions, normalize=normalize, prediction_field=prediction_field
    )


def extract_final_answer(example, completions, extract=None):
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")
    if extract:
        completions = [extract(example, p) for p in completions]
    else:
        completions = [
            p.strip().split("\n")[-1].split(":", 1)[-1].strip() for p in completions
        ]

    # TODO: make thread-safe?
    dsp.settings.lm.history.append(
        {**dsp.settings.lm.history[-1], "completions": completions}
    )

    return completions


def majority(
    completions: Completions, normalize: bool = True, field: Optional[str] = None
):
    """Returns the most common completion for the target field or the last field in the template."""
    field = completions.template.fields[-1].output_variable if field is None else field

    return Completions(
        majority_vote_(completions, normalize=normalize, prediction_field=field),
        template=completions.template,
    )


def majority_vote_(completions: Completions, normalize: bool, prediction_field: str):
    """Core logic for majority vote."""

    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")

    normalized_to_original = {}
    if normalize:
        original_completions = completions
        completions_ = []
        for pred in completions:
            if prediction_field in pred:
                completions_.append(normalize_text(pred[prediction_field]))
            else:
                completions_.append("")
        completions = completions_

        for completion, normalized_completion in zip(original_completions, completions):
            if normalized_completion not in normalized_to_original:
                normalized_to_original[normalized_completion] = completion

    completions_ = [x for x in completions if x]

    if completions_:
        completions = completions_

    topk = Counter(completions).most_common()
    pred, _ = topk[0]

    if normalize:
        pred = normalized_to_original[pred]

    dsp.settings.lm.history.append(
        {**dsp.settings.lm.history[-1], "topk": topk, "completions": [pred]}
    )

    return [pred]

from contextlib import contextmanager

@contextmanager
def dry_run(predict_instance):
    """
    A context manager for performing a dry run of the prediction process without actually generating predictions.

    This function temporarily replaces the `forward` method of the prediction instance with a mock version that simulates
    the generation process. It is useful for testing and debugging purposes, allowing the examination of the inputs and
    configurations that would be used in a real prediction call.

    Args:
        predict_instance: The prediction instance whose `forward` method is to be mocked.

    Yields:
        None: This function does not yield any value but allows the execution of code within its context.

    Raises:
        AssertionError: If no language model is loaded in the settings.
    """
    # Save the original generate method of the instance
    original_forward = predict_instance.forward

    def mock_generate(template, **kwargs):
        """A mock generate function that simulates the generation process."""
        if not dsp.settings.lm:
            raise AssertionError("No LM is loaded.")
        
        def _mock_do_generate(example, stage, max_depth=2, original_example=None):
            """Simulates the do_generate process without actual generation."""
            if not dsp.settings.lm:
                raise AssertionError("No LM is loaded.")
            original_example = original_example or example
            assert stage is not None, "Stage must be specified"

            # Simulate looking up the appropriate fields in each demonstration.
            example = example.demos_at(lambda d: d[stage])

            # Simulate generating and extracting the fields.
            prompt = template(example)

            return prompt, []

        return _mock_do_generate

    def mock_forward(self, **kwargs):
        """
        A mock version of the forward method that simulates the prediction process.

        This function extracts configuration parameters from the provided keyword arguments,
        simulates the generation process, and constructs a dictionary containing information
        about the dry run, including the generated prompt, configuration, and signature.
        """
        # Extract the three privileged keyword arguments.
        new_signature = kwargs.pop("new_signature", None)
        signature = kwargs.pop("signature", predict_instance.signature)
        demos = kwargs.pop("demos", predict_instance.demos)
        config = dict(**predict_instance.config, **kwargs.pop("config", {}))

        # Determine the language model to use.
        lm = kwargs.pop("lm", predict_instance.lm) or dsp.settings.lm

        # Adjust temperature based on the number of generations requested.
        temperature = config.get("temperature", None)
        temperature = lm.kwargs['temperature'] if temperature is None else temperature
        num_generations = config.get("n", None)
        if num_generations is None:
            num_generations = lm.kwargs.get('n', lm.kwargs.get('num_generations', None))
        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        # Simulate the generation process.
        x = dsp.Example(demos=demos, **kwargs)
        if new_signature is not None:
            signature = dsp.Template(signature.instructions, **new_signature)
        if predict_instance.lm is None:
            x, C = mock_generate(signature, **config)(x, stage=predict_instance.stage)
        else:
            with dsp.settings.context(lm=predict_instance.lm, query_only=True):
                x, C = mock_generate(signature, **config)(x, stage=predict_instance.stage)

        # Construct the dry run information.
        dry_run_info = {
            'prompt': x,
            'config': config,
            'signature': str(signature),
            'stage': predict_instance.stage,
        }

        # Encode the prompt if an encoder is available.
        encoder = dsp.settings.config.get('encoder', None)
        if encoder is not None:
            encoded_tokens = encoder.encode(x)
            dry_run_info['encoded_tokens'] = encoded_tokens
            dry_run_info['token_count'] = len(encoded_tokens)

        return dry_run_info

    # Temporarily replace the generate method of the instance with the mock_generate
    predict_instance.forward = mock_forward.__get__(predict_instance, predict_instance.__class__)
    try:
        # Yield control back to the with block, allowing the user to call predict_instance methods
        yield
    finally:
        # Restore the original generate method to the instance after exiting the with block
        predict_instance.forward = original_forward