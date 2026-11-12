"""`TrainingJob.cancel` returns whether the future was actually cancelled."""

from dspy.clients.provider import TrainingJob


class _Job(TrainingJob):
    def status(self):
        return "cancelled"


def test_cancel_returns_true_for_a_pending_job():
    assert _Job(model="test-model").cancel() is True


def test_cancel_returns_false_for_a_finished_job():
    job = _Job(model="test-model")
    job.set_result("done")
    assert job.cancel() is False
