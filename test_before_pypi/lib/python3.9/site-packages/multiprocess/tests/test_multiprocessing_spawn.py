import unittest
from multiprocess.tests import install_tests_in_module_dict

from test import support

if support.PGO:
    raise unittest.SkipTest("test is not helpful for PGO")

install_tests_in_module_dict(globals(), 'spawn')

if __name__ == '__main__':
    unittest.main()
