import typing as t
from dspy.signatures.signature import Signature
from dspy.primitives.example import Example


def passages_to_text(passages: t.Iterable[str]) -> str:
    assert len(passages) > 0

    return "\n".join([f"[{idx + 1}] <<{text}>>" for idx, text in enumerate(passages)])


def format_answers(answers: t.Iterable[str]) -> str:
    assert len(answers) > 0
    return (answers[0]).strip()


def default_format_handler(x: str) -> str:
    assert type(x) == str
    return " ".join(x.split())


class Template:
    def __init__(self, signature: Signature, **kwargs):
        self.signature = signature
        self.kwargs = kwargs

        self.format_handlers: dict[str, t.Callable] = {
            "context": passages_to_text,
            "passages": passages_to_text,
            "answers": format_answers,
        }

        for key, field in signature.fields.items():
            format = field.json_schema_extra.get("format")
            if format:
                self.format_handlers[key] = format

    def query(self, example: Example, is_demo: bool = False) -> str:
        result = []

        # Fill in Output Variable Values in Example and ensure all params are provided
        if not is_demo:
            for field in self.signature.fields:
                if field not in example:
                    example[field] = ""
                    break

        # Iterate through Fields
        for name, field in self.signature.fields.items():
            if name in self.format_handlers:
                format_handler = self.format_handlers[name]
            else:
                format_handler = default_format_handler

            formatted_value = format_handler(example[name])

            result.append(f"{name.capitalize()}: {formatted_value}")

        return "\n".join(result)

    def guidelines(self) -> str:
        """Returns the task guidelines as described in the lm prompt"""
        result = "Follow the following format.\n\n"

        example = Example()
        field_strings = []
        for name, field in self.signature.fields.items():
            field_strings.append(
                f"{name.capitalize()}: {field.json_schema_extra['desc']}"
            )

        return result + "\n".join(field_strings)

    def extract(self, example: Example, raw_pred: str) -> Example:
        """Extracts the answer from the LM raw prediction using the template structure

        Args:
            example (Example): Contains the input variables that raw_pred was completed on.
            raw_pred (str): LM generated field_strings

        Returns:
            Example: The example with the output variables filled in

        """

        output_fields = [
            (name, field)
            for name, field in self.signature.fields.items()
            if field.json_schema_extra["__dspy_field_type"] == "output"
        ]

        example[output_fields[0][0]] = raw_pred

        return example

    def __call__(self, example: Example, show_guidelines: bool = True) -> str:
        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(self.signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self.guidelines())

        # Generate Spans for Each Demo
        field_names = list(self.signature.fields.keys())
        for demo in example.demos:
            demo_example = Example()
            for idx, variable in enumerate(demo):
                print(idx, variable, field_names)
                name = field_names[idx]
                demo_example[name] = variable

            prompt_spans.append(self.query(demo_example))

        # Generate Empty Demo for Generation
        prompt_spans.append(self.query(example))

        return "\n\n---\n\n".join([span.strip() for span in prompt_spans])
