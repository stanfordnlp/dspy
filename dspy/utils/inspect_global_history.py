# TODO: Consider if this should access settings.lm *or* a list that's shared across all LMs in the program.
def inspect_history(*args, **kwargs):
    from dspy.clients.base_lm import GLOBAL_HISTORY, _inspect_history
    return _inspect_history(GLOBAL_HISTORY, *args, **kwargs)