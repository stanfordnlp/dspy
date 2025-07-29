from .predict import Predict


class Tool:
    """A simple tool class for ReAct."""
    
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ReAct(Predict):
    """A simplified ReAct implementation."""
    
    def __init__(self, signature, tools=None, **kwargs):
        super().__init__(signature, **kwargs)
        self.tools = tools or []
        
    def forward(self, **kwargs):
        # Simplified ReAct - just call the parent Predict
        return super().forward(**kwargs) 