import re
import dsp

from .field import Field
import threading

class SignatureMeta(type):
    _thread_local_storage = threading.local()

    class _SignatureNamespace:
        def __init__(self, fields):
            for key, value in fields.items():
                setattr(self, key, value)

    def __new__(cls, name, bases, class_dict):
        type_attributes = {}

        for k, v in list(class_dict.items()):
            if isinstance(v, Field):
                v.finalize(k, infer_prefix(k))
                type_attributes[k] = v
                del class_dict[k]

        instructions = class_dict.get('__doc__') or ""

        new_class = super().__new__(cls, name, bases, class_dict)

        # Attach the _SignatureNamespace directly to the class
        setattr(new_class, 'signature', cls._SignatureNamespace(type_attributes))

        # Create and attach the template directly to the class
        setattr(new_class, '_template', dsp.Template(instructions=instructions, **type_attributes))

        return new_class

    def __call__(cls, *args, **kwargs):
        return cls._template(*args, **kwargs)

    def __getattr__(cls, attr):
        # Redirect attribute access to the template object when accessed on the class directly
        return getattr(cls._template, attr)
    
    def __repr__(cls):
        s = []

        for name, field in cls.signature.__dict__.items():
            s.append(f"- {name} = {field}")
        
        return f'{cls.__name__}\n' + '\n'.join(s)

class Signature(metaclass=SignatureMeta):
    pass



def infer_prefix(attribute_name: str) -> str:
    """Infers a prefix from an attribute name."""
    
    # Convert camelCase to snake_case, but handle sequences of capital letters properly
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', attribute_name)
    intermediate_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)

    # Insert underscores around numbers to ensure spaces in the final output
    with_underscores_around_numbers = re.sub('([a-zA-Z])(\d)', r'\1_\2', intermediate_name)
    with_underscores_around_numbers = re.sub('(\d)([a-zA-Z])', r'\1_\2', with_underscores_around_numbers)

    # Convert snake_case to 'Proper Title Case', but ensure acronyms are uppercased
    words = with_underscores_around_numbers.split('_')
    title_cased_words = []
    for word in words:
        if word.isupper():
            title_cased_words.append(word)
        else:
            title_cased_words.append(word.capitalize())
    
    return ' '.join(title_cased_words)

### Testing the function
assert infer_prefix('someAttributeName42IsCool') == 'Some Attribute Name 42 Is Cool'
assert infer_prefix('version2Update') == 'Version 2 Update'
assert infer_prefix('modelT45Enhanced') == 'Model T 45 Enhanced'
assert infer_prefix('someAttributeName') == 'Some Attribute Name'
assert infer_prefix('some_attribute_name') == 'Some Attribute Name'
assert infer_prefix('URLAddress') == 'URL Address'
assert infer_prefix('isHTTPSecure') == 'Is HTTP Secure'
assert infer_prefix('isHTTPSSecure123') == 'Is HTTPS Secure 123'
