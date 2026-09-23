import pytest

import dspy
from tests.teleprompt.reanchor.fakes import FakeClient


@pytest.fixture
def system_one():
    """Installs a FakeClient as `dspy.settings.system_one` for one test."""

    def install(answer):
        client = FakeClient(answer)
        dspy.configure(system_one=client)
        return client

    return install
