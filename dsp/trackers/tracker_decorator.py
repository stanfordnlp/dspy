from dsp import BaseTracker


def tracker_decorator(func):
    def tracker_wrapper(*args, **kwargs):
        completions = func(*args, **kwargs)
        if hasattr(args[0], "tracker") and issubclass(args[0], BaseTracker):
            args[0].tracker.call(i=kwargs['prompt'], o=completions, **kwargs)
        return completions
    return tracker_wrapper
