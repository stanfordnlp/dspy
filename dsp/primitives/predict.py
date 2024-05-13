from collections import Counter
from typing import Any, Callable, Optional

import dsp
from dsp.primitives.demonstrate import Example
from dsp.templates.template_v3 import Template
from dsp.utils import normalize_text, zipstar
from dsp.utils.utils import dotdict


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


class UnsetBackendError(AssertionError):
    """Raised when no backend is configured."""


def _generate(template: Template, **kwargs) -> Callable:
    """Returns a callable function that generates completions for a given example using the provided template."""
    generator = dsp.settings.lm
    if not generator:
        raise UnsetBackendError("No backend configured.")

    def extend_generation(
        completion: Example, field_names: list[str], stage: str, max_depth: int, original_example: Example,
    ):
        """If the required fields are not present in the completion, extend the generation."""
        assert max_depth > 0, "Max depth exceeded - failed to complete in one pass - increase max_tokens"
        # remove content of last field to avoid half-completed content
        for field_name in get_all_fields_following_missing_field(completion, field_names):
            completion.pop(field_name, None)

        # Recurse with greedy decoding and a shorter length.
        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_output_tokens")
            or dsp.settings.lm.kwargs.get("max_tokens")
            or dsp.settings.lm.kwargs.get("max_output_tokens")
        )

        if max_tokens is None:
            raise ValueError("Required 'max_tokens' or 'max_output_tokens' not specified in settings.")
        max_tokens = min(max(75, max_tokens // 2), max_tokens)
        keys = list(kwargs.keys()) + list(dsp.settings.lm.kwargs.keys())
        max_tokens_key = "max_tokens" if "max_tokens" in keys else "max_output_tokens"
        new_kwargs = {
            **kwargs,
            max_tokens_key: max_tokens,
            "n": 1,
            "temperature": 0.0,
        }

        _, finished_completion = generate(template, **new_kwargs)(
            completion,
            stage=stage,
            max_depth=max_depth - 1,
            original_example=original_example,
        )
        return finished_completion.data[0]

    def do_generate(
        example: Example,
        stage: str,
        max_depth: int = 2,
        original_example=None,
    ):
        if not dsp.settings.lm:
            raise AssertionError("No LM or backend is loaded.")
        original_example = original_example or example
        assert stage is not None

        # Look up the appropriate fields in each demonstration.
        example = example.demos_at(lambda d: d[stage])

        # Generate and extract the fields.
        prompt = template(example)
        completions: list[dict[str, Any]] = generator(prompt, **kwargs)
        completions: list[Example] = [template.extract(example, p) for p in completions]

        # Find the completions that are unfinished.
        field_names: list[str] = [field.input_variable for field in template.fields]

        finished_completions = []
        for completion in completions:
            if all((completion.get(key, "") != "") for key in field_names):
                finished_completions.append(completion)
                continue
            finished_completions.append(
                extend_generation(completion, field_names, stage, max_depth, original_example),
            )

        completions = Completions(finished_completions, template=template)
        example = example.copy(completions=completions)

        if len(completions) == 1:
            completion = completions[0]
            example[stage] = example.copy(**completion)

            if dsp.settings.compiling:
                inputs_ = set(original_example.keys())
                inputs = [f.input_variable for f in template.fields if f.input_variable in inputs_]
                outputs = [f.output_variable for f in template.fields if f.input_variable not in inputs_]

                example.compiling_stages = example.get("compiling_stages", [])
                example.compiling_stages.append(
                    {
                        "name": stage,
                        "template": template,
                        "inputs": inputs,
                        "outputs": outputs,
                    },
                )
        else:
            # assert not dsp.settings.compiling, "TODO: At this point, cannot compile n>1 generations"
            example[stage] = dotdict(completions=completions)

        return example, completions

    return do_generate


def get_all_fields_following_missing_field(completion: Example, field_names: list[str]) -> list[str]:
    """Returns every field following the first missing field"""
    for i, field_name in enumerate(field_names):
        if field_name not in completion:
            return field_names[i:]
        if completion[field_name] == "":
            return field_names[i:]
    return []


def generate_sc(
    example,
    prompt,
    normalize=True,
    extract=None,
    prediction_field=None,
    **kwargs,
):
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")
    kwargs = {"temperature": 0.7, "n": 20, "max_tokens": 150, **kwargs}

    completions = dsp.settings.lm(prompt, **kwargs)
    completions = extract_final_answer(example, completions, extract=extract)
    return majority_vote_(
        completions,
        normalize=normalize,
        prediction_field=prediction_field,
    )


def extract_final_answer(example, completions, extract=None):
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")
    if extract:
        completions = [extract(example, p) for p in completions]
    else:
        completions = [p.strip().split("\n")[-1].split(":", 1)[-1].strip() for p in completions]

    # TODO: make thread-safe?
    dsp.settings.lm.history.append(
        {**dsp.settings.lm.history[-1], "completions": completions},
    )

    return completions


def majority(
    completions: Completions,
    normalize: bool = True,
    field: Optional[str] = None,
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
        {**dsp.settings.lm.history[-1], "topk": topk, "completions": [pred]},
    )

    return [pred]
