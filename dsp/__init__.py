from .modules import *  # noqa
from .primitives import *  # noqa
from .adapters import *  # noqa
from .utils import settings  # noqa

"""
TODO:

The DspModule class serves as a proxy to our original 'dsp' module. It provides direct access to settings 
stored in `dsp_settings` as if they were top-level attributes of the 'dsp' module, while also ensuring that
all other regular attributes (like functions, classes, or submodules) of the 'dsp' module remain accessible.

By replacing the module's symbols with an instance of DspModule, we allow users to access settings 
with the syntax `dsp.<setting_name>` instead of the longer `dsp.dsp_settings.<setting_name>`. This makes 
for more concise and intuitive code. However, due to its unconventional nature, developers should be 
careful when modifying this module to ensure they maintain the expected behavior and access patterns.
"""


"""

class DspModule:
    
    def __init__(self):
        # Import and store the original module object
        self._original_module = sys.modules[__name__]
    
    def __getattr__(self, name):
        # First, try getting the attribute from the original module
        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)
        
        # Next, check dsp_settings
        if hasattr(dsp_settings, name):
            return getattr(dsp_settings, name)
        
        raise AttributeError(f"'{type(self).__name__}' object and the original module have no attribute '{name}'")

import sys
sys.modules[__name__] = DspModule()

"""