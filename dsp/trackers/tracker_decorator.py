
def tracker(func):
    def tracker_wrapper(*args, **kwargs):
        completions = func(*args, **kwargs)
        try:
            from dsp import BaseTracker
            if hasattr(args[0], "tracker") and issubclass(args[0].tracker.__class__, BaseTracker):
                args[0].tracker.call(i=args[1], o=completions, name=args[0].__class__.__name__, **args[0].kwargs)
        except Exception as e:
            raise RuntimeError(f"tracker TypeError and tracker.call() fail, detail:{e}") from e
        return completions
    return tracker_wrapper
