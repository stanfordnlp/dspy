""" 
TODO: If we want to have Prediction::{**keys, completions, box} where box.{key} will behave as a value but also include
the completions internally.

The main thing left is to determine the semantic (and then implement them) for applying operations on the object.

If we have a string (query) and completions (5 queries), and we modify the string, what happens to the completions?

- Option 1: We modify the string and the (other) completions are not affected.
- Option 2: We modify the string and the (other) completions too.
- Option 3: We modify the string and the (other) completions are deleted.

Option 2 seems most reasonable, but it depends on what metadata the box is going to store.

It seems that a box fundamentally has two functions then:
- Store a value and its "alternatives" (and hence allow transparent application over operations on value/alts)
    - But not all operations can/should be applied on all as a map.
    - I guess mainly "give me a string or list or dict or tuple or int or float" has to commit to a value.
    - There also needs to be a .item().
- Potentially track the "source" of the value (i.e., the predictor that generated it, and its inputs)
- Give the value (eventually) to something that will consume the main value (implicitly or explicitly) or all/some of its alternatives explicitly.

It might be wise to make this responsible for a smaller scope for now:

- Just one string (and its alternatives).
- No source tracking.
- Allow operations on the string to map over the alternatives.
- Seamless extraction at code boundaries.
    - Basically, code will either treat this as string implicitly
        (and hence only know about the one value, and on best effort basis we update the alternatives)
    - Or code will explicitly work with the string or explicitly work with the full set of alternatives.

- By default, all programs (and their sub-programs) will be running inside a context in which preserve_boxes=True.
- But outside the program, once we see that none of the parent contexts have preserve_boxes=True, we can automatically
    unpack all boxes before returning to user.

Okay, so we'll have predictors return a `pred` in which `pred.query` is a box.

You'd usually do one of:

### Things that just give you one string
    1- Print `pred.query` or save it in a dict somewhere or a file somewhere.
    2- Call `pred.query.item()` to get the string explicitly.
    3- Modifications in freeform Python.
        - Modify it by calling `pred.query = 'new query'` altogether.
        - Modify it by doing `pred.query += 'new query'` or templating `f'{pred.query} new query'`.
        - Other modifications are not allowed on strings (e.g., `pred.query[0] = 'a'` or `pred.query[0] += 'a'`).
        - Cast to boolean after a comparison: `if pred.query == 'something': ...`
            - Pytorch would say RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            - But we can keep the primary value and use that in the boolean.
            - So technically, comparison can stick around, giving you multiple internal bools.

Overall, I think it's coherent semantics, for the time being, to say that any of the above will just give you a string back and lose all tracking.


### Things that give you a list of strings
    1- Explicitly asking for the candidates/completions.
    2- Then you could filter or map that list arbitrarily.

In this case, it's just that Box will serve as syntactic sugar. If you don't want to think about `n` at all, you can
pretend you have a string. If you do anything arbitrary on it, it indeed becomes a string.
If you later decide to treat it as a list, it's easy to do so without losing that info when you say `pred.query`.
    
### Things that are more interesting

A) You can now pass pred.query to a DSPy predictor (or searcher, etc) and it can either naively work with the string,
like pass it to a template, or it can explicitly ask for the list of candidates and do something with that.

This will need a lot more string-specific operations though:
- endswith, startswith, contains, split, strip, lower, upper, etc.
- when doing ' '.join() must do map(str, values_to_join). No implicit __str__ conversion!
- We can probably automate this by having a general fallback? That either returns one value or maps that over all of them.

B) When you say dspy.assert pred.sentence1.endswith('blue'), it will actually check all the alternatives and locally filter them if possible.
It may keep the bad ones somewhere just in case too.

We could make this a little more explicit like dspy.assert(pred.sentence1, lambda x: x.endswith('blue'))

C) When program_temperature is high, we can actually have some more interesting logic here. When you try to do things that are "selective",
maybe we'll randomly give you one of the strings (that remain valid in the box, based on assertions).

This could lead to interesting efficiency, because we can basically rerun the program, it'll still generate n=10 candidates,
but each time it'll use a different one. So when branch_index changes, you get a new candidate each time, but it should be consistent in the same box.
I guess this could be done by shuffling the same N=10 things. So basically, there's a user N and there's a system-level M.

We can sometimes optimize things by doing M=5. So we'll generate Nx5 candidates in one or more calls (depending on value of Nx5).
Then depending on the branch_idx, we'll return a fixed set of N candidates. But we have the rest.

"""


class BoxType(type):
    # List of operations to override
    ops = [
        # Arithmetic operations
        'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 
        'lshift', 'rshift', 'and', 'or', 'xor',
        # 'r'-prefixed versions of arithmetic operations
        'radd', 'rsub', 'rmul', 'rtruediv', 'rfloordiv', 'rmod', 
        'rpow', 'rlshift', 'rrshift', 'rand', 'ror', 'rxor',
        # Sequence operations
        'getitem', 'setitem', 'delitem', 'contains',
        # Unary and other operations
        'neg', 'pos', 'abs', 'invert', 'round', 'len', 
        'getitem', 'setitem', 'delitem', 'contains', 'iter',
        # Mappings operations (for dicts)
        'get', 'keys', 'values', 'items',
        # Comparison
        'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    ]

    def __init__(cls, name, bases, attrs):
        def create_method(op):
            def method(self, other=None):
                if op in ['len', 'keys', 'values', 'items']:
                    return getattr(self._value, op)()
                elif isinstance(other, Box):
                    return Box(getattr(self._value, f'__{op}__')(other._value))
                elif other is not None:
                    return Box(getattr(self._value, f'__{op}__')(other))
                else:
                    return NotImplemented
            return method

        for op in BoxType.ops:
            setattr(cls, f'__{op}__', create_method(op))

        super().__init__(name, bases, attrs)


class Box(metaclass=BoxType):
    def __init__(self, value, source=False):
        self._value = value
        self._source = source

    def __repr__(self):
        return repr(self._value)

    def __str__(self):
        return str(self._value)
    
    def __bool__(self):
        return bool(self._value)
    
    # if method is missing just call it on the _value
    def __getattr__(self, name):
        return Box(getattr(self._value, name))

    # # Unlike the others, this one collapses to a bool directly
    # def __eq__(self, other):
    #     if isinstance(other, Box):
    #         return self._value == other._value
    #     else:
    #         return self._value == other

    # def __ne__(self, other):
    #     return not self.__eq__(other)
