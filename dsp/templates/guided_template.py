from pydantic.fields import FieldInfo
from typing import Callable, Any, Union
from dsp.templates import Template, format_answers, passages2text
from collections import namedtuple
from dsp.primitives.demonstrate import Example
import dsp

from guidance import gen 
GuidedField = namedtuple("Field", "name separator input_variable output_variable description pattern is_input")


def convert_pydantic_field(field_name: str, field: FieldInfo):
    prefix: str = field.json_schema_extra["prefix"]
    desc: str = field.json_schema_extra["desc"]
    pattern: str = next((x.pattern for x in field.metadata if hasattr(x ,'pattern')), None)
    is_input: bool = field.json_schema_extra["__dspy_field_type"] == "input"
    separator: str = (
        " " if prefix.rstrip() == prefix and len(prefix) > 0 else prefix[len(prefix.rstrip()) :]
    )
    return GuidedField(
        name=prefix.strip(),
        description=desc,
        input_variable=field_name,
        output_variable=field_name,
        separator=separator,
        is_input=is_input,
        pattern=pattern)

class GuidedTemplate(Template):
    def __init__(self, instructions: str, pydantic_fields: dict[str, FieldInfo], **kwargs) -> None:
        self.instructions = instructions
        self.kwargs = kwargs
        self.fields: list[GuidedField] = []
        self.format_handlers: dict[str, Callable] = {
            "context": passages2text,
            "passages": passages2text,
            "answers": format_answers,
        }

        for name, field in pydantic_fields.items():
            self.fields.append(convert_pydantic_field(name, field))
            if field.json_schema_extra.get("format"):
                self.format_handlers[name] = field.json_schema_extra["format"]
        


    def query(self, example: Example, is_demo: bool = False) -> str:
        """Retrieves the input variables from the example and formats them into a query string."""
        result: list[str] = []
        example = Example(example)

        def in_example(field: GuidedField) -> bool:
            return (field.input_variable in example 
                    and example[field.input_variable] is not None 
                    and example[field.input_variable] != "")
        
        def is_output(field: GuidedField) -> bool:
            return field.is_input == False
        
        if not is_demo:
            for field in self.fields:
                if not in_example(field) and is_output(field):
                    if field.pattern:
                        gencall = gen(regex=rf"{field.pattern}", name=field.input_variable)
                        example[field.input_variable] = f"{gencall}"
                    else:
                        gencall = gen(name=field.input_variable, max_tokens=100, stop='\n')
                        example[field.input_variable] = f"{gencall}"
        
        for field in self.fields:
            in_example = field.input_variable in example

            if in_example:
                value = example[field.input_variable]

            if (
                field.input_variable in example
                and example[field.input_variable] is not None
            ):
                if field.input_variable in self.format_handlers:
                    format_handler = self.format_handlers[field.input_variable]
                else:
                    def format_handler(x):
                        assert type(x) == str, f"Need format_handler for {field.input_variable} of type {type(x)}"
                        return " ".join(x.split())

                formatted_value = format_handler(example[field.input_variable])
                separator = '\n' if field.separator == ' ' and '\n' in formatted_value else field.separator

                result.append(
                    f"{field.name}{separator}{formatted_value}",
                )

        if self._has_augmented_guidelines() and (example.get('augmented', False)):
            return "\n\n".join([r for r in result if r])
        return "\n".join([r for r in result if r])
    
    def __call__(self, example, show_guidelines=True) -> str:
        example = Example(example)

        if hasattr(dsp.settings, 'query_only') and dsp.settings.query_only:
            return self.query(example)

        # The training data should not contain the output variable
        if self.fields[-1].input_variable in example:
            del example[self.fields[-1].input_variable]

        rdemos = [
            self.query(demo, is_demo=True)
            for demo in example.demos
            if (
                (not demo.get('augmented', False))
                and (  # validate that the training example has the same primitive input var as the template
                    self.fields[-1].input_variable in demo
                    and demo[self.fields[-1].input_variable] is not None
                )
            )
        ]

        ademos = [
            self.query(demo, is_demo=True)
            for demo in example.demos
            if demo.get('augmented', False)
        ]

        # Move the rdemos to ademos if rdemo has all the fields filled in
        rdemos_ = []
        new_ademos = []
        for rdemo in rdemos:
            if all(
                (field.name in rdemo)
                for field in self.fields
                if field.input_variable in example
            ):
                import dspy

                if dspy.settings.release >= 20230928:
                    new_ademos.append(rdemo)
                else:
                    ademos.append(rdemo)
            else:
                rdemos_.append(rdemo)

        ademos = new_ademos + ademos
        rdemos = rdemos_


        long_query = self._has_augmented_guidelines()

        if long_query:
            example["augmented"] = True

        query = self.query(example)

        # if it has more lines than fields
        if len(query.split('\n')) > len(self.fields):
            long_query = True

            if not example.get('augmented', False):
                example["augmented"] = True
                query = self.query(example)

        rdemos = "\n\n".join(rdemos)

        if len(rdemos) >= 1 and len(ademos) == 0 and not long_query:
            rdemos_and_query = "\n\n".join([rdemos, query])
            parts = [
                self.instructions,
                self.guidelines(show_guidelines),
                rdemos_and_query,
            ]
        elif len(rdemos) == 0:
            parts = [
                self.instructions,
                self.guidelines(show_guidelines),
                *ademos,
                query,
            ]
        else:
            parts = [
                self.instructions,
                rdemos,
                self.guidelines(show_guidelines),
                *ademos,
                query,
            ]

        prompt = "\n\n---\n\n".join([p.strip() for p in parts if p])

        return prompt.strip()
    
    def extract(
        self, example: Union[Example, dict[str, Any]], raw_pred: dict[str, Any]
    ) -> Example:
        """Extracts the answer from the LM raw prediction using the template structure

        Args:
            example (Union[Example, dict[str, Any]]): Contains the input variables that raw_pred was completed on.
            raw_pred dict[str,Any]: Guided LM generated key/value pairs 

        Returns:
            Example: The example with the output variables filled in
        """
        example = Example(example)
        for key, value in raw_pred.items():
            example[key] = value

        return example
