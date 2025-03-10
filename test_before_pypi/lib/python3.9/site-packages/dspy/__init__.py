from dspy.predict import *
from dspy.primitives import *
from dspy.retrieve import *
from dspy.signatures import *
from dspy.teleprompt import *

import dspy.retrievers

from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, Image, History  # isort: skip
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.asyncify import asyncify
from dspy.utils.saving import load
from dspy.utils.streaming import streamify

from dspy.dsp.utils.settings import settings

configure_dspy_loggers(__name__)

from dspy.dsp.colbertv2 import ColBERTv2
# from dspy.dsp.you import You

configure = settings.configure
context = settings.context

BootstrapRS = BootstrapFewShotWithRandomSearch

from .__metadata__ import __name__, __version__, __description__, __url__, __author__, __author_email__
