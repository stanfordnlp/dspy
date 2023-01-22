from dsp.templates import TemplateV2, passages2text, format_answers, Field

class Type:
    def __init__(self, prefix, desc, format=None, aliases=[]) -> None:
        self.prefix = prefix
        self.desc = desc
        self.format = format
        self.aliases = aliases

        assert aliases == [], "TODO: If aliases is used, handling of input_variable in V2 needs to be richer."
    
    def __call__(self, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return Type(**kwargs)

class Template(TemplateV2):
    def __init__(self, instructions, **kwargs):
        self.instructions = instructions

        self.fields = []
        self.format_handlers = {"context": passages2text, "answers": format_answers}

        for key, value in kwargs.items():
            prefix = value.prefix
            separator = ' ' if prefix.rstrip() == prefix else prefix[len(prefix.rstrip()):]
            field = Field(name=prefix.strip(), description=value.desc,
                          input_variable=key, output_variable=key, seperator=separator)
            self.fields.append(field)

            if value.format:
                for name in [key, *value.aliases]:
                    self.format_handlers[name] = value.format
