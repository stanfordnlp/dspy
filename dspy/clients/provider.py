from abc import abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import TYPE_CHECKING, Any

from dspy.clients.utils_finetune import TrainDataFormat

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class TrainingJob(Future):
    """Represents an asynchronous fine-tuning job for a language model.

    TrainingJob extends `concurrent.futures.Future` to provide a standard interface
    for tracking and managing LM fine-tuning operations. Subclasses should implement
    provider-specific logic for job management.

    Attributes:
        thread: The thread running the training job, if applicable.
        model: The name/identifier of the model being fine-tuned.
        train_data: The training data used for fine-tuning.
        train_data_format: The format of the training data.
        train_kwargs: Additional keyword arguments for training configuration.
    """

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
    """Represents a reinforcement learning training job for a language model.

    ReinforceJob provides an interface for iterative RL-based training with checkpoint
    management. Subclasses should implement provider-specific logic for initialization,
    training steps, and checkpoint operations.

    Attributes:
        lm: The language model instance being trained.
        train_kwargs: Additional keyword arguments for training configuration.
        checkpoints: Dictionary mapping checkpoint names to checkpoint data.
        last_checkpoint: The name of the most recently saved checkpoint.
    """

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
    def save_checkpoint(self, checkpoint_name: str):
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class Provider:
    """Base class for LM provider implementations.

    Provider defines the interface for interacting with different LM backends,
    including model launching, fine-tuning, and reinforcement learning capabilities.
    Subclasses should implement provider-specific logic and set capability flags.

    Attributes:
        finetunable: Whether this provider supports fine-tuning.
        reinforceable: Whether this provider supports reinforcement learning.
        TrainingJob: The TrainingJob class to use for fine-tuning jobs.
        ReinforceJob: The ReinforceJob class to use for RL jobs.
    """

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
