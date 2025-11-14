import pytest

from dspy.utils import magicattr


class Test:
    l = [1, 2]
    a = [0, [1, 2, [3, 4]]]
    b = {"x": {"y": "y"}, "z": [1, 2]}
    z = "z"


class Person:
    settings = {
        "autosave": True,
        "style": {"height": 30, "width": 200},
        "themes": ["light", "dark"],
    }

    def __init__(self, name, age, friends):
        self.name = name
        self.age = age
        self.friends = friends


@pytest.mark.parametrize(
    "key, value",
    [
        ("l", Test.l),
        ("t.t.t.t.z", "z"),
        ("a[0]", 0),
        ("a[1][0]", 1),
        ("a[1][2]", [3, 4]),
        ('b["x"]', {"y": "y"}),
        ('b["x"]["y"]', "y"),
        ('b["z"]', [1, 2]),
        ('b["z"][1]', 2),
        ('b["w"].z', "z"),
        ('b["w"].t.l', [1, 2]),
        ("a[-1].z", "z"),
        ("l[-1]", 2),
        ("a[2].t.a[-1].z", "z"),
        ('a[2].t.b["z"][0]', 1),
        ("a[-1].t.z", "z"),
    ],
)
def test_magicattr_get(key, value):
    obj = Test()
    obj.t = obj
    obj.a.append(obj)
    obj.b["w"] = obj

    assert magicattr.get(obj, key) == value


def test_person_example():
    bob = Person(name="Bob", age=31, friends=[])
    jill = Person(name="Jill", age=29, friends=[bob])
    jack = Person(name="Jack", age=28, friends=[bob, jill])

    assert magicattr.get(bob, "age") == 31

    with pytest.raises(AttributeError):
        magicattr.get(bob, "weight")
    assert magicattr.get(bob, "weight", default=75) == 75

    assert magicattr.get(jill, "friends[0].name") == "Bob"
    assert magicattr.get(jack, "friends[-1].age") == 29

    assert magicattr.get(jack, 'settings["style"]["width"]') == 200

    assert magicattr.get(jack, 'settings["themes"][-2]') == "light"
    assert magicattr.get(jack, 'friends[-1].settings["themes"][1]') == "dark"

    magicattr.set(bob, 'settings["style"]["width"]', 400)
    assert magicattr.get(bob, 'settings["style"]["width"]') == 400

    magicattr.set(bob, "friends", [jack, jill])
    assert magicattr.get(jack, "friends[0].friends[0]") == jack

    magicattr.set(jill, "friends[0].age", 32)
    assert bob.age == 32

    magicattr.delete(jill, "friends[0]")
    assert len(jill.friends) == 0

    magicattr.delete(jill, "age")
    assert not hasattr(jill, "age")

    magicattr.delete(bob, "friends[0].age")
    assert not hasattr(jack, "age")

    with pytest.raises(NotImplementedError):
        magicattr.get(bob, "friends[0+1]")

    with pytest.raises(ValueError):
        magicattr.get(bob, "friends.pop(0)")

    with pytest.raises(ValueError):
        magicattr.get(bob, "friends = []")

    with pytest.raises(SyntaxError):
        magicattr.get(bob, "friends..")

    with pytest.raises(KeyError):
        magicattr.get(bob, 'settings["DoesNotExist"]')

    with pytest.raises(IndexError):
        magicattr.get(bob, "friends[100]")


def test_empty():
    obj = Test()
    with pytest.raises(ValueError):
        magicattr.get(obj, "   ")

    with pytest.raises(ValueError):
        magicattr.get(obj, "")

    with pytest.raises(TypeError):
        magicattr.get(obj, 0)

    with pytest.raises(TypeError):
        magicattr.get(obj, None)

    with pytest.raises(TypeError):
        magicattr.get(obj, obj)
