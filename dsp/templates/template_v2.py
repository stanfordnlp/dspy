from collections import namedtuple
import re

import dsp

Field = namedtuple("Field", "name seperator input_variable output_variable description")

# TODO: de-duplicate with dsp/templates/template.py


def passages2text(passages):
    if type(passages) is str:
        return passages

    assert type(passages) in [list, tuple]

    if len(passages) == 1:
        return f"«{passages[0]}»"

    return "\n".join([f"[{idx+1}] «{txt}»" for idx, txt in enumerate(passages)])


def format_answers(answers):
    if isinstance(answers, list):
        if len(answers) >= 1:
            return str(answers[0]).strip()
        elif len(answers) == 0:
            raise ValueError(f"No answers found")
    elif isinstance(answers, str):
        return answers
    else:
        raise ValueError(f"Unable to parse answers of type {type(answers)}")


class TemplateV2:
    def __init__(self, template, format_handlers={"passages": passages2text, "context": passages2text, "answer": format_answers, "answers": format_answers}):
        self.format_handlers = format_handlers

        template = template.strip()

        self.instructions = re.search("(.*)\n", template).group(1)
        template = template[len(self.instructions):].strip()

        self.fields = []
        while len(template) > 0:
            match = re.search("(.*)(\s){(.*)}\s(.*\${.*})", template)
            if match is not None:
                name = match.group(1)
                seperator = match.group(2)
                variable = match.group(3)
                description = match.group(4)
            else:
                match = re.search("(.*)(\s){(.*)}", template)
                if match is not None:
                    name = match.group(1)
                    seperator = match.group(2)
                    variable = match.group(3)
                    description = None
                else:
                    raise ValueError(f"Could not parse template")

            var_match = re.match("(.*) -> (.*)", variable)
            if var_match is not None:
                input_variable = var_match.group(1)
                output_variable = var_match.group(2)
            else:
                input_variable = variable
                output_variable = variable

            self.fields.append(
                Field(
                    name=name,
                    seperator=seperator,
                    input_variable=input_variable,
                    output_variable=output_variable,
                    description=description,
                )
            )

            template = template[len(match.group(0)):].strip()

    def query(self, example, is_demo=False):
        result = []

        if not is_demo:
            has_value = [
                field.input_variable in example
                and example[field.input_variable] is not None
                and example[field.input_variable] != ""
                for field in self.fields
            ]

            for i in range(1, len(has_value)):
                if has_value[i - 1] and not any(has_value[i:]):
                    example[self.fields[i].input_variable] = ""
                    break

        for field in self.fields:
            if (field.input_variable in example and example[field.input_variable] is not None):
                if field.input_variable in self.format_handlers:
                    format_handler = self.format_handlers[field.input_variable]
                else:
                    def format_handler(x): return ' '.join(x.split())
                
                result.append(f"{field.name}{field.seperator}{format_handler(example[field.input_variable])}")

        if self._has_augmented_guidelines() and ("augmented" in example and example.augmented):
            return "\n\n".join(result)
        else:
            return "\n".join(result)

    def guidelines(self):
        result = "Follow the following format.\n\n"

        example = dsp.Example()
        for field in self.fields:
            example[field.input_variable] = field.description
        example.augmented = self._has_augmented_guidelines()

        result += self.query(example)
        return result

    def _has_augmented_guidelines(self):
        return len(self.fields) > 3 or any(field.seperator == "\n" for field in self.fields)

    def extract(self, example, raw_pred):
        example = dsp.Example(example)

        raw_pred = raw_pred.strip()

        idx = 0
        while idx < len(self.fields):
            if (self.fields[idx].input_variable not in example or example[self.fields[idx].input_variable] is None):
                break
            idx += 1
        
        idx = min(idx, len(self.fields) - 1)

        while raw_pred != "" and idx < len(self.fields):
            # print("HERE,", idx)
            if idx < len(self.fields) - 1:
                next_field_name = '\n' + self.fields[idx + 1].name
                offset = raw_pred.find(next_field_name)
                # print(f"Looking for ```{next_field_name}```. Found it at {offset}")

                if offset >= 0:
                    example[self.fields[idx].output_variable] = raw_pred[:offset].strip()
                    raw_pred = raw_pred[offset + len(next_field_name):].strip()
                    idx += 1
                else:
                    example[self.fields[idx].output_variable] = raw_pred.strip()
                    raw_pred = ""
                    idx += 1
                    # print("Exiting")
                    break

            else: #if idx == len(self.fields) - 1:
                assert idx == len(self.fields) - 1, (idx, len(self.fields))
                example[self.fields[idx].output_variable] = raw_pred.strip()
                break
                # offset = raw_pred.find(self.fields[idx].name)
                # if offset < 0:
                # else:
                #     raw_pred = raw_pred[offset + len(self.fields[idx].name):].strip()
                #     example[self.fields[idx].output_variable] = raw_pred

        return example

    def __call__(self, example):
        example = dsp.Example(example)

        if self.fields[-1].input_variable in example:
            del example[self.fields[-1].input_variable]

        rdemos = [
            self.query(demo, is_demo=True)
            for demo in example.demos
            if (
                ("augmented" not in demo or not demo.augmented)
                and (
                    self.fields[-1].input_variable in demo
                    and demo[self.fields[-1].input_variable] is not None
                )
            )
        ]
        ademos = [
            self.query(demo, is_demo=True)
            for demo in example.demos
            if "augmented" in demo and demo.augmented
        ]

        long_query = self._has_augmented_guidelines()
        if long_query:
            example["augmented"] = True
        query = self.query(example)

        rdemos = "\n\n".join(rdemos)
        if len(rdemos) >= 1 and len(ademos) == 0 and not long_query:
            rdemos_and_query = "\n\n".join([rdemos, query])
            parts = [self.instructions, self.guidelines(), rdemos_and_query]
        elif len(rdemos) == 0:
            parts = [self.instructions, self.guidelines(), *ademos, query]
        else:
            parts = [self.instructions, rdemos,
                     self.guidelines(), *ademos, query]

        prompt = "\n\n---\n\n".join([p.strip() for p in parts if p])

        return prompt.strip()
