"""Static contracts for the experimental decorator, checked by ty rather than executed by pytest.

Deliberately invalid calls use ignores to assert expected diagnostics. Unused-ignore
errors make these checks fail if the decorator stops rejecting invalid calls.
"""

from typing_extensions import assert_type

from dspy.utils.annotation import experimental


@experimental
def compute(count: int, /, *, prefix: str) -> str:
    return f"{prefix}: {count}"


@experimental(version="3.5.0")
def versioned_greet(name: str) -> int:
    return len(name)


@experimental
class Service:
    def __init__(self, port: int) -> None:
        self.port = port

    def ping(self) -> bool:
        return True


@experimental(version="2.0.0")
class Config:
    endpoint: str = "https://example.com"


class Container:
    @property
    @experimental
    def capacity(self) -> int:
        return 100


def check_experimental_contracts() -> None:
    # 1. Unversioned function decoration
    assert_type(compute(3, prefix="items"), str)
    compute("three", prefix="items")  # ty: ignore[invalid-argument-type]
    compute(3)  # ty: ignore[missing-argument]
    compute(3, "items", prefix="items")  # ty: ignore[too-many-positional-arguments]
    compute(count=3, prefix="items")  # ty: ignore[positional-only-parameter-as-kwarg]

    # 2. Versioned function decoration
    assert_type(versioned_greet("Alice"), int)
    versioned_greet(123)  # ty: ignore[invalid-argument-type]
    versioned_greet()  # ty: ignore[missing-argument]

    # 3. Class decoration preserves constructor signature and instance types
    svc = Service(8080)
    assert_type(svc, Service)
    assert_type(svc.port, int)
    assert_type(svc.ping(), bool)
    Service("not-a-port")  # ty: ignore[invalid-argument-type]
    Service()  # ty: ignore[missing-argument]

    # 4. Versioned class decoration preserves attributes
    cfg = Config()
    assert_type(cfg, Config)
    assert_type(cfg.endpoint, str)

    # 5. Direct invocation syntax preserves signature
    def raw_helper(flag: bool) -> list[int]:
        return [1] if flag else []

    wrapped_helper = experimental(raw_helper)
    assert_type(wrapped_helper(True), list[int])
    wrapped_helper("not-a-bool")  # ty: ignore[invalid-argument-type]

    wrapped_versioned = experimental(raw_helper, version="4.0.0")
    assert_type(wrapped_versioned(False), list[int])

    # 6. Invalid version argument types are rejected
    experimental(version=123)  # ty: ignore[invalid-argument-type]
    experimental(raw_helper, version=123)  # ty: ignore[no-matching-overload]

    # 7. Property decoration preserves property return type
    container = Container()
    assert_type(container.capacity, int)
