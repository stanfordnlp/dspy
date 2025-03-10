#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

import dill
from enum import EnumMeta
import sys
dill.settings['recurse'] = True

# test classdefs
class _class:
    def _method(self):
        pass
    def ok(self):
        return True

class _class2:
    def __call__(self):
        pass
    def ok(self):
        return True

class _newclass(object):
    def _method(self):
        pass
    def ok(self):
        return True

class _newclass2(object):
    def __call__(self):
        pass
    def ok(self):
        return True

class _meta(type):
    pass

def __call__(self):
    pass
def ok(self):
    return True

_mclass = _meta("_mclass", (object,), {"__call__": __call__, "ok": ok})

del __call__
del ok

o = _class()
oc = _class2()
n = _newclass()
nc = _newclass2()
m = _mclass()

if sys.hexversion < 0x03090000:
    import typing
    class customIntList(typing.List[int]):
        pass
else:
    class customIntList(list[int]):
        pass

# test pickles for class instances
def test_class_instances():
    assert dill.pickles(o)
    assert dill.pickles(oc)
    assert dill.pickles(n)
    assert dill.pickles(nc)
    assert dill.pickles(m)

def test_class_objects():
    clslist = [_class,_class2,_newclass,_newclass2,_mclass]
    objlist = [o,oc,n,nc,m]
    _clslist = [dill.dumps(obj) for obj in clslist]
    _objlist = [dill.dumps(obj) for obj in objlist]

    for obj in clslist:
        globals().pop(obj.__name__)
    del clslist
    for obj in ['o','oc','n','nc']:
        globals().pop(obj)
    del objlist
    del obj

    for obj,cls in zip(_objlist,_clslist):
        _cls = dill.loads(cls)
        _obj = dill.loads(obj)
        assert _obj.ok()
        assert _cls.ok(_cls())
        if _cls.__name__ == "_mclass":
            assert type(_cls).__name__ == "_meta"

# test NoneType
def test_specialtypes():
    assert dill.pickles(type(None))
    assert dill.pickles(type(NotImplemented))
    assert dill.pickles(type(Ellipsis))
    assert dill.pickles(type(EnumMeta))

from collections import namedtuple
Z = namedtuple("Z", ['a','b'])
Zi = Z(0,1)
X = namedtuple("Y", ['a','b'])
X.__name__ = "X"
X.__qualname__ = "X" #XXX: name must 'match' or fails to pickle
Xi = X(0,1)
Bad = namedtuple("FakeName", ['a','b'])
Badi = Bad(0,1)
Defaults = namedtuple('Defaults', ['x', 'y'], defaults=[1])
Defaultsi = Defaults(2)

# test namedtuple
def test_namedtuple():
    assert Z is dill.loads(dill.dumps(Z))
    assert Zi == dill.loads(dill.dumps(Zi))
    assert X is dill.loads(dill.dumps(X))
    assert Xi == dill.loads(dill.dumps(Xi))
    assert Defaults is dill.loads(dill.dumps(Defaults))
    assert Defaultsi == dill.loads(dill.dumps(Defaultsi))
    assert Bad is not dill.loads(dill.dumps(Bad))
    assert Bad._fields == dill.loads(dill.dumps(Bad))._fields
    assert tuple(Badi) == tuple(dill.loads(dill.dumps(Badi)))

    class A:
        class B(namedtuple("C", ["one", "two"])):
            '''docstring'''
        B.__module__ = 'testing'

    a = A()
    assert dill.copy(a)

    assert dill.copy(A.B).__name__ == 'B'
    assert dill.copy(A.B).__qualname__.endswith('.<locals>.A.B')
    assert dill.copy(A.B).__doc__ == 'docstring'
    assert dill.copy(A.B).__module__ == 'testing'

    from typing import NamedTuple

    def A():
        class B(NamedTuple):
            x: int
        return B

    assert type(dill.copy(A()(8))).__qualname__ == type(A()(8)).__qualname__

def test_dtype():
    try:
        import numpy as np

        dti = np.dtype('int')
        assert np.dtype == dill.copy(np.dtype)
        assert dti == dill.copy(dti)
    except ImportError: pass


def test_array_nested():
    try:
        import numpy as np

        x = np.array([1])
        y = (x,)
        assert y == dill.copy(y)

    except ImportError: pass


def test_array_subclass():
    try:
        import numpy as np

        class TestArray(np.ndarray):
            def __new__(cls, input_array, color):
                obj = np.asarray(input_array).view(cls)
                obj.color = color
                return obj
            def __array_finalize__(self, obj):
                if obj is None:
                    return
                if isinstance(obj, type(self)):
                    self.color = obj.color
            def __getnewargs__(self):
                return np.asarray(self), self.color

        a1 = TestArray(np.zeros(100), color='green')
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a1)
            assert a1.__dict__ == dill.copy(a1).__dict__

        a2 = a1[0:9]
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a2)
            assert a2.__dict__ == dill.copy(a2).__dict__

        class TestArray2(np.ndarray):
            color = 'blue'

        a3 = TestArray2([1,2,3,4,5])
        a3.color = 'green'
        if not dill._dill.IS_PYPY:
            assert dill.pickles(a3)
            assert a3.__dict__ == dill.copy(a3).__dict__

    except ImportError: pass


def test_method_decorator():
    class A(object):
      @classmethod
      def test(cls):
        pass

    a = A()

    res = dill.dumps(a)
    new_obj = dill.loads(res)
    new_obj.__class__.test()

# test slots
class Y(object):
  __slots__ = ('y', '__weakref__')
  def __init__(self, y):
    self.y = y

value = 123
y = Y(value)

class Y2(object):
  __slots__ = 'y'
  def __init__(self, y):
    self.y = y

def test_slots():
    assert dill.pickles(Y)
    assert dill.pickles(y)
    assert dill.pickles(Y.y)
    assert dill.copy(y).y == value
    assert dill.copy(Y2(value)).y == value

def test_origbases():
    assert dill.copy(customIntList).__orig_bases__ == customIntList.__orig_bases__

def test_attr():
    import attr
    @attr.s
    class A:
        a = attr.ib()

    v = A(1)
    assert dill.copy(v) == v

def test_metaclass():
    class metaclass_with_new(type):
        def __new__(mcls, name, bases, ns, **kwds):
            cls = super().__new__(mcls, name, bases, ns, **kwds)
            assert mcls is not None
            assert cls.method(mcls)
            return cls
        def method(cls, mcls):
            return isinstance(cls, mcls)

    l = locals()
    exec("""class subclass_with_new(metaclass=metaclass_with_new):
        def __new__(cls):
            self = super().__new__(cls)
            return self""", None, l)
    subclass_with_new = l['subclass_with_new']

    assert dill.copy(subclass_with_new())

def test_enummeta():
    from http import HTTPStatus
    import enum
    assert dill.copy(HTTPStatus.OK) is HTTPStatus.OK
    assert dill.copy(enum.EnumMeta) is enum.EnumMeta

def test_inherit(): #NOTE: see issue #612
    class Foo:
        w = 0
        x = 1
        y = 1.1
        a = ()
        b = (1,)
        n = None

    class Bar(Foo):
        w = 2
        x = 1
        y = 1.1
        z = 0.2
        a = ()
        b = (1,)
        c = (2,)
        n = None

    Baz = dill.copy(Bar)

    import platform
    is_pypy = platform.python_implementation() == 'PyPy'
    assert Bar.__dict__ == Baz.__dict__
    # ints
    assert 'w' in Bar.__dict__ and 'w' in Baz.__dict__
    assert Bar.__dict__['w'] is Baz.__dict__['w']
    assert 'x' in Bar.__dict__ and 'x' in Baz.__dict__
    assert Bar.__dict__['x'] is Baz.__dict__['x']
    # floats
    assert 'y' in Bar.__dict__ and 'y' in Baz.__dict__
    same = Bar.__dict__['y'] is Baz.__dict__['y']
    assert same if is_pypy else not same
    assert 'z' in Bar.__dict__ and 'z' in Baz.__dict__
    same = Bar.__dict__['z'] is Baz.__dict__['z']
    assert same if is_pypy else not same
    # tuples
    assert 'a' in Bar.__dict__ and 'a' in Baz.__dict__
    assert Bar.__dict__['a'] is Baz.__dict__['a']
    assert 'b' in Bar.__dict__ and 'b' in Baz.__dict__
    assert Bar.__dict__['b'] is not Baz.__dict__['b']
    assert 'c' in Bar.__dict__ and 'c' in Baz.__dict__
    assert Bar.__dict__['c'] is not Baz.__dict__['c']
    # None
    assert 'n' in Bar.__dict__ and 'n' in Baz.__dict__
    assert Bar.__dict__['n'] is Baz.__dict__['n']


if __name__ == '__main__':
    test_class_instances()
    test_class_objects()
    test_specialtypes()
    test_namedtuple()
    test_dtype()
    test_array_nested()
    test_array_subclass()
    test_method_decorator()
    test_slots()
    test_origbases()
    test_metaclass()
    test_enummeta()
    test_inherit()
