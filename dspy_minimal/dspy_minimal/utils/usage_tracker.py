from contextlib import contextmanager


class UsageTracker:
    """Simple usage tracker for LM calls."""
    
    def __init__(self):
        self.total_tokens = 0
        
    def add_usage(self, model, usage):
        """Add usage information."""
        if usage and "total_tokens" in usage:
            self.total_tokens += usage["total_tokens"]
            
    def get_total_tokens(self):
        """Get total tokens used."""
        return self.total_tokens


@contextmanager
def track_usage():
    """Context manager for tracking usage."""
    tracker = UsageTracker()
    try:
        yield tracker
    finally:
        pass 