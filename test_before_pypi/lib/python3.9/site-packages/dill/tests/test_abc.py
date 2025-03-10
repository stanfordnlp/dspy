#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2023-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
test dill's ability to pickle abstract base class objects
"""
import dill
import abc
from abc import ABC
import warnings

from types import FunctionType

dill.settings['recurse'] = True

class OneTwoThree(ABC):
    @abc.abstractmethod
    def foo(self):
        """A method"""
        pass

    @property
    @abc.abstractmethod
    def bar(self):
        """Property getter"""
        pass

    @bar.setter
    @abc.abstractmethod
    def bar(self, value):
        """Property setter"""
        pass

    @classmethod
    @abc.abstractmethod
    def cfoo(cls):
        """Class method"""
        pass

    @staticmethod
    @abc.abstractmethod
    def sfoo():
        """Static method"""
        pass

class EasyAsAbc(OneTwoThree):
    def __init__(self):
        self._bar = None

    def foo(self):
        return "Instance Method FOO"

    @property
    def bar(self):
        return self._bar

    @bar.setter
    def bar(self, value):
        self._bar = value

    @classmethod
    def cfoo(cls):
        return "Class Method CFOO"

    @staticmethod
    def sfoo():
        return "Static Method SFOO"

def test_abc_non_local():
    assert dill.copy(OneTwoThree) is not OneTwoThree
    assert dill.copy(EasyAsAbc) is not EasyAsAbc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dill.PicklingWarning)
        assert dill.copy(OneTwoThree, byref=True) is OneTwoThree
        assert dill.copy(EasyAsAbc, byref=True) is EasyAsAbc

    instance = EasyAsAbc()
    # Set a property that StockPickle can't preserve
    instance.bar = lambda x: x**2
    depickled = dill.copy(instance)
    assert type(depickled) is type(instance) #NOTE: issue #612, test_abc_local
    #NOTE: dill.copy of local (or non-local) classes should (not) be the same?
    assert type(depickled.bar) is FunctionType
    assert depickled.bar(3) == 9
    assert depickled.sfoo() == "Static Method SFOO"
    assert depickled.cfoo() == "Class Method CFOO"
    assert depickled.foo() == "Instance Method FOO"

def test_abc_local():
    """
    Test using locally scoped ABC class
    """
    class LocalABC(ABC):
        @abc.abstractmethod
        def foo(self):
            pass

        def baz(self):
            return repr(self)

    labc = dill.copy(LocalABC)
    assert labc is not LocalABC
    assert type(labc) is type(LocalABC)
    #NOTE: dill.copy of local (or non-local) classes should (not) be the same?
    # <class '__main__.LocalABC'>
    # <class '__main__.test_abc_local.<locals>.LocalABC'>

    class Real(labc):
        def foo(self):
            return "True!"

        def baz(self):
            return "My " + super(Real, self).baz()

    real = Real()
    assert real.foo() == "True!"

    try:
        labc()
    except TypeError as e:
        # Expected error
        pass
    else:
        print('Failed to raise type error')
        assert False

    labc2, pik = dill.copy((labc, Real()))
    assert 'Real' == type(pik).__name__
    assert '.Real' in type(pik).__qualname__
    assert type(pik) is not Real
    assert labc2 is not LocalABC
    assert labc2 is not labc
    assert isinstance(pik, labc2)
    assert not isinstance(pik, labc)
    assert not isinstance(pik, LocalABC)
    assert pik.baz() == "My " + repr(pik)

def test_meta_local_no_cache():
    """
    Test calling metaclass and cache registration
    """
    LocalMetaABC = abc.ABCMeta('LocalMetaABC', (), {})

    class ClassyClass:
        pass

    class KlassyClass:
      pass

    LocalMetaABC.register(ClassyClass)

    assert not issubclass(KlassyClass, LocalMetaABC)
    assert issubclass(ClassyClass, LocalMetaABC)

    res = dill.dumps((LocalMetaABC, ClassyClass, KlassyClass))

    lmabc, cc, kc = dill.loads(res)
    assert type(lmabc) == type(LocalMetaABC)
    assert not issubclass(kc, lmabc)
    assert issubclass(cc, lmabc)

if __name__ == '__main__':
    test_abc_non_local()
    test_abc_local()
    test_meta_local_no_cache()
