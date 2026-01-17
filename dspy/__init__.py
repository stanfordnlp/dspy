import warnings

# Suppress warnings from upstream dependencies that users can't act on.
# These are all issues in openai/litellm internals that don't affect dspy functionality.

# openai 1.x imports from pydantic.v1 for typing utilities that still work correctly.
# Fixed in openai >=2.0.0 (Sep 30, 2025).
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

# litellm's logging calls model_dump() on response objects with a Choices/StreamingChoices
# schema mismatch. Fixed in litellm >=1.80.16 (Jan 13, 2026).
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
)

# litellm uses asyncio.iscoroutinefunction which is deprecated in Python 3.14+.
warnings.filterwarnings(
    "ignore",
    message="'asyncio.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
)

# litellm uses deprecated Pydantic class-based config.
warnings.filterwarnings(
    "ignore",
    message="Support for class-based `config` is deprecated",
    category=DeprecationWarning,
)

from dspy.predict import *
from dspy.primitives import *
from dspy.retrievers import *
from dspy.signatures import *
from dspy.teleprompt import *

from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter, Image, Audio, File, History, Type, Tool, ToolCalls, Code, Reasoning  # isort: skip
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.asyncify import asyncify
from dspy.utils.syncify import syncify
from dspy.utils.saving import load
from dspy.streaming.streamify import streamify
from dspy.utils.usage_tracker import track_usage

from dspy.dsp.utils.settings import settings
from dspy.dsp.colbertv2 import ColBERTv2
from dspy.clients import DSPY_CACHE
from dspy.__metadata__ import __name__, __version__, __description__, __url__, __author__, __author_email__

configure_dspy_loggers(__name__)

# Singleton definitions and aliasing
configure = settings.configure
context = settings.context

BootstrapRS = BootstrapFewShotWithRandomSearch

cache = DSPY_CACHE
