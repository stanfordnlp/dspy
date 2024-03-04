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

    def guidelines(self) -> str:
        """Returns the task guidelines as described in the lm prompt"""
        result = "Follow the following format.\n\n"

        field_strings = []
        for field in self.signature.fields.values():
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

        full_text = self.__call__(example) + raw_pred

        if not full_text.endswith("\n\n---"):
            full_text = full_text + "\n\n---"

        # Generate Search Strings
        search_strings = []
        output_fields = list(self.signature.output_fields.keys())
        for idx, (key, field) in enumerate(self.signature.output_fields.items()):
            if len(search_strings) > 0:
                search_strings[-1] += f"{field.json_schema_extra['prefix']}"

            target_str = f"(?s){field.json_schema_extra['prefix']}\\s(.+?)"
            if idx != len(self.signature.output_fields) - 1:
                target_str += "\\n\\n"
            else:
                target_str += "\\n\\n\\-\\-\\-"

            search_strings.append(target_str)

        # Generate Results
        if len(self.signature.output_fields) == 1:
            matches = regex.findall(search_strings[0], full_text)

            # If no matches are found, and there are is only one prediction, return entire prediction
            if matches is None:
                example[output_fields[0]] = full_text
            else:
                example[output_fields[0]] = matches[-1]

        else:
            count = None
            for idx, field in enumerate(output_fields):
                matches = regex.findall(search_strings[idx], raw_pred)
                if count is not None and len(matches) != count:
                    break

                count = len(matches)

                print(matches)

                if len(matches) > 0:
                    print(matches[-1])
                    example[field] = matches[-1]

        return example

    def __call__(self, example: Example, show_guidelines: bool = True) -> str:
        prompt_spans = []

        # Start by getting the instructions
        prompt_spans.append(self.signature.instructions)

        # Generate the Guidelines
        prompt_spans.append(self.guidelines())

        # Generate Spans for Each Demo
        for demo in example.get("demos", []):
            prompt_spans.append(self.query(demo, is_demo=True))

        # Generate Empty Demo for Generation
        prompt_spans.append(self.query(example, is_demo=False))

        return "\n\n---\n\n".join([span.strip() for span in prompt_spans])
