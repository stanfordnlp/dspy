import re
import dsp

class Field:
    """A more ergonomic datatype that infers prefix and desc if omitted."""
    def __init__(self, *, prefix=None, desc=None, input, format=None):
        self.prefix = prefix  # This can be None initially and set later
        self.desc = desc
        self.format = format
        
    def finalize(self, key, inferred_prefix):
        """Set the prefix if it's not provided explicitly."""
        if self.prefix is None:
            self.prefix = inferred_prefix + ":"
        
        if self.desc is None:
            self.desc = f'${{{key}}}'
        
    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.prefix}, desc={self.desc})"
    
    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

class InputField(Field):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)

class OutputField(Field):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
