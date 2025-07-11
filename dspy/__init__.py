from dspy.predict import *
from dspy.primitives import *
from dspy.retrievers import *
from dspy.signatures import *
from dspy.teleprompt import *

from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter, Image, Audio, History, Type, Tool, ToolCalls, Code  # isort: skip
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
