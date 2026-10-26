import hashlib

from dspy.utils.hasher import Hasher


def test_hasher_hash_bytes_single():
    data = b"test payload"
    expected = hashlib.sha256(data).hexdigest()
    assert Hasher.hash_bytes(data) == expected


def test_hasher_hash_bytes_list():
    chunks = [b"hello", b" ", b"world"]
    expected = hashlib.sha256(b"".join(chunks)).hexdigest()
    assert Hasher.hash_bytes(chunks) == expected


def test_hasher_hash_determinism():
    obj1 = {"query": "sample query", "demos": [1, 2, 3]}
    obj2 = {"query": "sample query", "demos": [1, 2, 3]}
    assert Hasher.hash(obj1) == Hasher.hash(obj2)


def test_hasher_hash_different_objects():
    hash1 = Hasher.hash("value_a")
    hash2 = Hasher.hash("value_b")
    assert hash1 != hash2


def test_hasher_update_and_hexdigest():
    h1 = Hasher()
    h1.update("first")
    h1.update(123)
    digest1 = h1.hexdigest()

    h2 = Hasher()
    h2.update("first")
    h2.update(123)
    digest2 = h2.hexdigest()

    assert digest1 == digest2

    h3 = Hasher()
    h3.update("first")
    h3.update(456)
    assert h3.hexdigest() != digest1


def test_hasher_dispatch_dict():
    assert isinstance(Hasher.dispatch, dict)
