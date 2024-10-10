from typing import Any, Union

import dsp
from dsp.primitives.demonstrate import Example

from .base_template import BaseTemplate


class ExperimentalAdapter(BaseTemplate):
    def query(self, example: Example, is_demo: bool = False) -> str:
        """Retrieves the input variables from the example and formats them into a query string."""
        result: list[str] = []

        # If not a demo, find the last field that doesn't have a value set in `example` and set it to ""
        # This creates the "Output:" prefix at the end of the prompt.
        if not is_demo:
            has_value = [
                field.input_variable in example
                and example[field.input_variable] is not None
                and example[field.input_variable] != ""
                for field in self.fields
            ]

            if not any(has_value):
                assert False, "No input variables found in the example"

            for i in range(1, len(has_value)):
                if has_value[i - 1] and not any(has_value[i:]):
                    example[self.fields[i].input_variable] = ""
                    break

        for field in self.fields:
            if field.input_variable in example and example[field.input_variable] is not None:
                if field.input_variable in self.format_handlers:
                    format_handler = self.format_handlers[field.input_variable]
                else:
                    def format_handler(x):
                        return str(x).strip()

                formatted_value = format_handler(example[field.input_variable])
                separator = "\n" if field.separator == " " and "\n" in formatted_value else field.separator

                result.append(f"{field.name}{separator}{formatted_value}",)

        return "\n\n".join([r for r in result if r])

    def guidelines(self, show_guidelines=True) -> str:
        """Returns the task guidelines as described in the lm prompt"""
        if (not show_guidelines) or (hasattr(dsp.settings, "show_guidelines") and not dsp.settings.show_guidelines):
            return ""

        result = "Follow the following format.\n\n"

        example = dsp.Example()
        for field in self.fields:
            example[field.input_variable] = field.description
        example.augmented = True

        result += self.query(example)
        return result

    def extract(
        self,
        example: Union[Example, dict[str, Any]],
        raw_pred: str,
    ) -> Example:
        """Extracts the answer from the LM raw prediction using the template structure

        Args:
            example (Union[Example, dict[str, Any]]): Contains the input variables that raw_pred was completed on.
            raw_pred (str): LM generated string

        Returns:
            Example: The example with the output variables filled in
        """
        example = dsp.Example(example)

        idx = 0
        while idx < len(self.fields):
            if self.fields[idx].input_variable not in example or example[self.fields[idx].input_variable] is None:
                break
            idx += 1

        import dspy

        idx = min(idx, len(self.fields) - 1)
        while raw_pred != "" and idx < len(self.fields):
            if idx < len(self.fields) - 1:
                next_field_name = "\n" + self.fields[idx + 1].name
                offset = raw_pred.find(next_field_name)

                if offset >= 0:
                    if dspy.settings.release >= 20231003:
                        example[self.fields[idx].output_variable] = raw_pred[:offset].strip().rstrip("---").strip()
                        raw_pred = raw_pred[offset + len(next_field_name) :].strip().rstrip("---").strip()
                    else:
                        field_name_parts = self.fields[idx].name.split()
                        start_pos = 0
                        for part in field_name_parts:
                            pos = raw_pred.find(part.strip())
                            if pos != -1:
                                start_pos = pos + len(part)
                            else:
                                break
                
                    example[self.fields[idx].output_variable] = raw_pred[start_pos:offset].strip().rstrip("---").strip()
                    raw_pred = raw_pred[offset + len(next_field_name) :].strip()
                    idx += 1
                else:
                    example[self.fields[idx].output_variable] = raw_pred.strip().rstrip("---").strip()

                    raw_pred = ""
                    idx += 1
                    break

            else:
                assert idx == len(self.fields) - 1, (idx, len(self.fields))

                if dspy.settings.release >= 20231003:
                    example[self.fields[idx].output_variable] = raw_pred.strip().rstrip("---").strip()
                else:
                    field_name_parts = self.fields[idx].name.split()
                    start_pos = 0
                    for part in field_name_parts:
                        pos = raw_pred.find(part.strip())
                        if pos != -1:
                            start_pos = pos + len(part)
                        else:
                            break
                    example[self.fields[idx].output_variable] = raw_pred[start_pos:].strip()

                break

        return example

    def __call__(self, example, show_guidelines=True) -> str:
        example = dsp.Example(example)
        output_fields = []
        for i in range(len(self.fields)):
            if self.fields[i].input_variable not in example:
                output_field = self.fields[i].input_variable
                if output_field not in output_fields:
                    output_fields.append(self.fields[i].name.split(':')[0])

        if hasattr(dsp.settings, "query_only") and dsp.settings.query_only:
            return self.query(example)

        # The training data should not contain the output variable
        assert self.fields[-1].input_variable not in example, f"Output variable {self.fields[-1].input_variable} should not be supplied for querying the LM."
        # del example[self.fields[-1].input_variable]

        rdemos = [
            self.query(demo, is_demo=True)
            for demo in example.demos
            if (
                (not demo.get("augmented", False))
                and (  # validate that the training example has the same primitive input var as the template
                    self.fields[-1].input_variable in demo and demo[self.fields[-1].input_variable] is not None
                )
            )
        ]

        ademos = [self.query(demo, is_demo=True) for demo in example.demos if demo.get("augmented", False)]

        # Move the rdemos to ademos if rdemo has all the fields filled in
        rdemos_ = []
        new_ademos = []
        for rdemo in rdemos:
            if all((field.name in rdemo) for field in self.fields if field.input_variable in example):
                new_ademos.append(rdemo)
            else:
                rdemos_.append(rdemo)

        ademos = new_ademos + ademos
        rdemos = rdemos_

        example["augmented"] = True

        query = self.query(example)
        parts = [self.instructions, *rdemos, self.guidelines(show_guidelines), *ademos, query,]

        prompt = "\n\n---\n\n".join([p.strip() for p in parts if p])
        prompt_ = prompt[: prompt.rfind("\n")].strip()

        s_or_not = "s" if len(output_fields) > 1 else ""
        only_or_not = "only " if len(output_fields) == 1 else ""

        prompt_ += f"\n\nPlease provide the output field{s_or_not} {', '.join(output_fields[:-1]) + (', then ' if len(output_fields) > 2 else ' then ') + output_fields[-1] if len(output_fields) > 1 else output_fields[0]}. Do so immediately, without additional content before or after, and precisely as the format above shows. Begin with {only_or_not}the field {output_fields[0]}."
        return prompt_.strip()

