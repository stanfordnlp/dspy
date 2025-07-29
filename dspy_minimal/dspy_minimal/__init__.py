# Minimal DSPy implementation for Lambda deployment
# Includes: Module, Predict, ReAct, Tool, Signature, InputField, OutputField, BootstrapFewShot, Example, LM, configure
# Plus: Adapter, Evaluate, exceptions, MCP support

from .primitives import Module, Example
from .predict import Predict, ReAct, Tool
from .signatures import Signature, InputField, OutputField
from .teleprompt import BootstrapFewShot
from .clients import LM
from .utils.settings import settings
from .adapters import Adapter
from .evaluate import Evaluate
from .utils.exceptions import AdapterParseError, DSPyError, SignatureError, ConfigurationError, LMError
from .mcp import ClientSession, StdioServerParameters, stdio_client

# Export the main components
__all__ = [
    "Module",
    "Predict", 
    "ReAct",
    "Tool",
    "Signature",
    "InputField",
    "OutputField",
    "BootstrapFewShot",
    "Example",
    "LM",
    "configure",
    "Adapter",
    "Evaluate",
    "AdapterParseError",
    "DSPyError",
    "SignatureError", 
    "ConfigurationError",
    "LMError",
    "ClientSession",
    "StdioServerParameters", 
    "stdio_client"
]

# Alias for configure function
configure = settings.configure 