import os
import sys

import sphinx

# Set the root path of the project
sys.path.insert(0, os.path.abspath('../'))

# Specify the path to the master document
master_doc = 'index'

# Set the project information
project = 'DSPy'
author = 'DSPy Team'
version = sphinx.__display_version__

# Add the extensions that Sphinx should use
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
