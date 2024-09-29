# from functools import wraps

# import dsp


# # applied right to left (innermost first, like function calls)
# def compose_decorators(*decorators):
#     def decorator(func):
#         for decorator in decorators[::-1]:
#             func = decorator(func)
#         return func
#     return decorator


# def shallow_copy_example_args(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         args = [dsp.Example(arg) if isinstance(arg, dsp.Example) else arg for arg in args]
#         kwargs = {key: dsp.Example(value) if isinstance(value, dsp.Example) else value for key, value in kwargs.items()}
#         return func(*args, **kwargs)
#     return wrapper


# transformation = shallow_copy_example_args
# # transformation = compose_decorators(handle_compilation, shallow_copy_example_args)



# def compiled(func):
#     def wrapper(*args, **kwargs):
#         is_to_be_compiled = True #decorator_kwargs.get('compile', False)
#         compiled_lm = dsp.settings.compiled_lm

#         if is_to_be_compiled and compiled_lm:
#             assert len(args) == 1, len(args)
#             example = args[0]

#             with dsp.settings.context(lm=compiled_lm, show_guidelines=False):
#                 old_demos = list(example.demos)
#                 example = func(example.copy(demos=[]), **kwargs)
#                 return example.copy(demos=old_demos)
        
#         with dsp.settings.context(compiling=True):
#             return func(*args, **kwargs)

#     return wrapper
