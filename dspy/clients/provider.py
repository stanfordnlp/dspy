from abc import abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import TYPE_CHECKING, Any

from dspy.clients.utils_finetune import TrainDataFormat

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class TrainingJob(Future):
    def __init__(
        self,
        thread: Thread | None = None,
        model: str | None = None,
        train_data: list[dict[str, Any]] | None = None,
        train_data_format: TrainDataFormat | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ):
        self.thread = thread
        self.model = model
        self.train_data = train_data
        self.train_data_format = train_data_format
        self.train_kwargs = train_kwargs or {}
        super().__init__()

    # Subclasses should override the cancel method to cancel the job; then call
    # the super's cancel method so that the future can be cancelled.
    def cancel(self):
        super().cancel()

    @abstractmethod
    def status(self):
        raise NotImplementedError


class ReinforceJob:
    def __init__(self, lm: "LM", train_kwargs: dict[str, Any] | None = None):
        self.lm = lm
        self.train_kwargs = train_kwargs or {}
        self.checkpoints = {}
        self.last_checkpoint = None

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | str | None = None):
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        raise NotImplementedError

    @abstractmethod
    def update_model(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str):
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class Provider:
    def __init__(self):
        self.finetunable = False
        self.reinforceable = False
        self.TrainingJob = TrainingJob
        self.ReinforceJob = ReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        # Subclasses should actually check whether a model is supported if they
        # want to have the model provider auto-discovered.
        return False

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        # Note that "launch" and "kill" methods might be called even if there
        # is a launched LM or no launched LM to kill. These methods should be
        # resillient to such cases.
        pass

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        # We assume that LM.launch_kwargs dictionary will contain the necessary
        # information for a provider to launch and/or kill an LM. This is the
        # reeason why the argument here is named launch_kwargs and not
        # kill_kwargs.
        pass

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError
