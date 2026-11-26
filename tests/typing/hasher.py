"""Static contracts for Hasher, checked by ty rather than executed by pytest.

Deliberately invalid calls use ignores to assert expected diagnostics. Unused-ignore
errors make these checks fail if the class stops rejecting invalid calls.
"""

from typing_extensions import assert_type

from dspy.utils.hasher import Hasher


def check_hasher_contracts() -> None:
    # 1. Hasher.hash accepts any pickle-serializable object and returns hex digest str
    assert_type(Hasher.hash("hello"), str)
    assert_type(Hasher.hash(123), str)
    assert_type(Hasher.hash({"key": "val"}), str)
    assert_type(Hasher.hash([1, 2, 3]), str)

    # 2. Hasher.hash_bytes accepts bytes or list[bytes] and returns hex digest str
    assert_type(Hasher.hash_bytes(b"chunk"), str)
    assert_type(Hasher.hash_bytes([b"chunk1", b"chunk2"]), str)

    # Negative argument types for hash_bytes
    Hasher.hash_bytes("not-bytes")  # ty: ignore[invalid-argument-type]
    Hasher.hash_bytes(123)  # ty: ignore[invalid-argument-type]
    Hasher.hash_bytes([123])  # ty: ignore[invalid-argument-type]

    # 3. Hasher instance methods
    hasher = Hasher()
    assert_type(hasher, Hasher)
    assert_type(hasher.update("payload"), None)
    assert_type(hasher.update(456), None)
    assert_type(hasher.update([b"data"]), None)
    assert_type(hasher.hexdigest(), str)

    # Negative call cases
    hasher.update()  # ty: ignore[missing-argument]
    hasher.hexdigest("extra")  # ty: ignore[too-many-positional-arguments]
