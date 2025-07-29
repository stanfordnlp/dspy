import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional


class Settings:
    """Global settings for DSPy."""
    
    def __init__(self):
        self.lm = None
        self.trace = None
        self.caller_modules = None
        self.track_usage = False
        self.usage_tracker = None
        
    def configure(self, **kwargs):
        """Configure global settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown setting: {key}")
                
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary settings."""
        old_settings = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_settings[key] = getattr(self, key)
                setattr(self, key, value)
        
        try:
            yield
        finally:
            for key, value in old_settings.items():
                setattr(self, key, value)


# Global settings instance
settings = Settings()

# Thread-local overrides
thread_local_overrides = threading.local()


def get_thread_local_overrides():
    """Get thread-local overrides."""
    if not hasattr(thread_local_overrides, 'overrides'):
        thread_local_overrides.overrides = {}
    return thread_local_overrides.overrides 