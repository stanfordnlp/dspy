import pytest

import dspy
from tests.teleprompt.reanchor.fakes import FakeClient


@pytest.fixture
def system_one():
    """Configures a FakeClient as the LM for one test."""

    def install(answer, cache=True):
        client = FakeClient(answer, cache=cache)
        dspy.configure(lm=client)
        return client

    return install
