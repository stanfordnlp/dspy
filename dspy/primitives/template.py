import regex
import typing as t
from dspy.signatures.signature import Signature
from dspy.primitives.example import Example


def passages_to_text(passages: t.Iterable[str]) -> str:
    assert len(passages) > 0
    if len(passages) > 1:
        return "\n".join(
            [f"[{idx + 1}] <<{text}>>" for idx, text in enumerate(passages)]
        )
    else:
        return passages[0]


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

    def _get_format_handler(self, name: str) -> t.Callable[[str], str]:
        if name in self.format_handlers:
            return self.format_handlers[name]

        return default_format_handler

    def _example_has_input_fields(self, example: Example):
        for name in self.signature.input_fields:
            if name not in example:
                raise Exception(f"Example missing necessary input field: {name}")

    def _example_has_output_fields(self, example: Example):
        for name in self.signature.output_fields:
            if name not in example:
                raise Exception(f"Example missing necessary output field: {name}")

    def query(self, example: Example, is_demo: bool) -> str:
        if is_demo:
            self._example_has_input_fields(example)
            self._example_has_output_fields(example)

        result = []

        # Append all Input Values, Regardless of Demo or not
        for name, field in self.signature.input_fields.items():
            format_handler = self._get_format_handler(name)

            result.append(
                f"{field.json_schema_extra['prefix']} {format_handler(example[name])}"
            )

        for name, field in self.signature.output_fields.items():
            format_handler = self._get_format_handler(name)

            if name not in example:
                result.append(f"{field.json_schema_extra['prefix']} ")
                break
            else:
                result.append(
                    f"{field.json_schema_extra['prefix']} {format_handler(example[name])}"
                )

        return "\n\n".join(result)

    def guidelines(self, is_json: bool = False) -> str:
        """Returns the task guidelines as described in the lm prompt"""
        if is_json:
            result = "Return the following fields in JSON format.\n\n"
        else:
            result = "Follow the following format.\n\n"

        field_strings = []
        for name, field in self.signature.fields.items():
            if is_json:
                field_strings.append(f"{name}: {field.json_schema_extra['desc']}")
            else:
                field_strings.append(
                    f"{field.json_schema_extra['prefix']} {field.json_schema_extra['desc']}"
                )

        return result + "\n\n".join(field_strings)

    def extract(self, example: Example, raw_pred: str) -> Example:
        """Extracts the answer from the LM raw prediction using the template structure

        Args:
            example (Example): Contains the input variables that raw_pred was completed on.
            raw_pred (str): LM generated field_strings

        Returns:
            Example: The example with the output variables filled in

        """

        # We have to deepcopy, so that the values dont continously overwrite each other.
        example = example.copy()

        full_text = self.__call__(example) + raw_pred

        if not full_text.endswith("\n\n---"):
            full_text = full_text + "\n\n---"

        print(f"FULL TEXT: {full_text}")

        # Generate Search Strings
        search_strings = []
        output_fields = list(self.signature.output_fields.keys())
        for idx, (key, field) in enumerate(self.signature.output_fields.items()):
            if len(search_strings) > 0:
                search_strings[-1] += f"(?:{field.json_schema_extra['prefix']})?"

            target_str = f"(?s)\n\n{field.json_schema_extra['prefix']}\\s?(.+?)"
            if idx != len(self.signature.output_fields) - 1:
                target_str += "\\n\\n"
            else:
                target_str += "\\n\\n\\-\\-\\-"

            search_strings.append(target_str)

        # Generate Results
        if len(self.signature.output_fields) == 1:
            matches = regex.findall(search_strings[0], full_text)

            # Skip the first match, as this is likely the demo.
            if len(matches) == 1:
                matches = []
            elif len(matches) > 1:
                matches = matches[1 + len(example.get("demos", [])) :]

            # If no matches are found, and there are is only one prediction, return entire prediction
            if matches == []:
                example[output_fields[0]] = full_text
            else:
                example[output_fields[0]] = matches[-1]

        else:
            for idx, field in enumerate(output_fields):
                matches = regex.findall(search_strings[idx], full_text)

                if len(matches) == 1:
                    matches = []
                elif len(matches) > 1:
                    matches = matches[1 + len(example.get("demos", [])) :]

                if len(matches) > 0:
                    example[field] = matches[-1]

        return example

    def __call__(
        self, example: Example, show_guidelines: bool = True, is_json: bool = False
    ) -> str:
        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(self.signature.instructions)

        # Generate the Guidelines
        if show_guidelines:
            prompt_spans.append(self.guidelines(is_json=is_json))

        # Generate Spans for Each Demo
        for demo in example.get("demos", []):
            prompt_spans.append(self.query(demo, is_demo=True))

        # Generate Empty Demo for Generation
        prompt_spans.append(self.query(example, is_demo=False))

        return "\n\n---\n\n".join([span.strip() for span in prompt_spans])
