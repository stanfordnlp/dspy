#
# Package analogous to 'threading.py' but using processes
#
# multiprocessing/__init__.py
#
# This package is intended to duplicate the functionality (and much of
# the API) of threading.py but uses processes instead of threads.  A
# subpackage 'multiprocessing.dummy' has the same API but is a simple
# wrapper for 'threading'.
#
# Original: Copyright (c) 2006-2008, R Oudkerk
# Original: Licensed to PSF under a Contributor Agreement.
# Forked by Mike McKerns, to support enhanced serialization.

# author, version, license, and long description
try: # the package is installed
    from .__info__ import __version__, __author__, __doc__, __license__
except: # pragma: no cover
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    sys.path.append(root)
    # get distribution meta info 
    from version import (__version__, __author__,
                         get_license_text, get_readme_as_rst)
    __license__ = get_license_text(os.path.join(root, 'LICENSE'))
    __license__ = "\n%s" % __license__
    __doc__ = get_readme_as_rst(os.path.join(root, 'README.md'))
    del os, sys, root, get_license_text, get_readme_as_rst


import sys
from . import context

#
# Copy stuff from default context
#

__all__ = [x for x in dir(context._default_context) if not x.startswith('_')]
globals().update((name, getattr(context._default_context, name)) for name in __all__)

#
# XXX These should not really be documented or public.
#

SUBDEBUG = 5
SUBWARNING = 25

#
# Alias for main module -- will be reset by bootstrapping child processes
#

if '__main__' in sys.modules:
    sys.modules['__mp_main__'] = sys.modules['__main__']


def license():
    """print license"""
    print (__license__)
    return

def citation():
    """print citation"""
    print (__doc__[-491:-118])
    return

