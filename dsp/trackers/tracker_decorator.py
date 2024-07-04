
def tracker_decorator(func):
    def tracker_wrapper(*args, **kwargs):
        completions = func(*args, **kwargs)
        from dsp import BaseTracker
        if hasattr(args[0], "tracker") and issubclass(args[0], BaseTracker):
            args[0].tracker.call(i=args[1], o=completions, **kwargs)
        return completions
    return tracker_wrapper
