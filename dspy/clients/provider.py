from abc import abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import TYPE_CHECKING, Any

from dspy.clients.utils_finetune import TrainDataFormat

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class TrainingJob(Future):
    """A future representing an asynchronous model fine-tuning job.

    This class extends `concurrent.futures.Future` to represent a fine-tuning job
    that can be monitored and cancelled. Subclasses should implement the `status`
    method to provide job status information specific to their provider.

    Attributes:
        thread: The thread running the training job, if applicable.
        model: The model identifier being fine-tuned.
        train_data: The training data used for fine-tuning.
        train_data_format: The format of the training data.
        train_kwargs: Additional keyword arguments for training configuration.

    Example:
        ```python
        from dspy.clients.provider import TrainingJob

        job = TrainingJob(
            model="gpt-3.5-turbo",
            train_data=[{"messages": [...]}],
            train_data_format=TrainDataFormat.CHAT
        )
        # Check if job is done
        if job.done():
            result = job.result()
        ```
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

    def cancel(self):
        """Cancel the training job.

        Subclasses should override this method to cancel the job with the provider;
        then call the super's cancel method so that the future can be cancelled.

        Returns:
            True if the job was successfully cancelled, False otherwise.
        """
        super().cancel()

    @abstractmethod
    def status(self):
        """Get the current status of the training job.

        This method should be implemented by subclasses to return the current
        status of the training job in a format specific to the provider.

        Returns:
            The current status of the training job. The exact type depends on
            the provider implementation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class ReinforceJob:
    """A job for reinforcement learning-based fine-tuning.

    This class represents a reinforcement learning fine-tuning job that can be
    initialized, stepped through training iterations, and terminated. It supports
    checkpointing to save progress during training.

    Attributes:
        lm: The language model being fine-tuned.
        train_kwargs: Additional keyword arguments for training configuration.
        checkpoints: Dictionary storing checkpoint data.
        last_checkpoint: The name of the most recent checkpoint.

    Example:
        ```python
        from dspy.clients.provider import ReinforceJob

        job = ReinforceJob(lm=my_lm, train_kwargs={"learning_rate": 0.001})
        job.initialize()
        job.step(train_data=[...], train_data_format=TrainDataFormat.CHAT)
        job.save_checkpoint("checkpoint_1")
        job.terminate()
        ```
    """

    def __init__(self, lm: "LM", train_kwargs: dict[str, Any] | None = None):
        self.lm = lm
        self.train_kwargs = train_kwargs or {}
        self.checkpoints = {}
        self.last_checkpoint = None

    @abstractmethod
    def initialize(self):
        """Initialize the reinforcement learning job.

        This method should be implemented by subclasses to set up the necessary
        resources and state for the reinforcement learning training process.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | str | None = None):
        """Perform one step of reinforcement learning training.

        This method should be implemented by subclasses to execute one iteration
        of the reinforcement learning training process using the provided training data.

        Args:
            train_data: A list of training examples, each represented as a dictionary.
            train_data_format: The format of the training data. Can be a `TrainDataFormat`
                enum value or a string identifier.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        """Terminate the reinforcement learning job.

        This method should be implemented by subclasses to clean up resources
        and finalize the training process.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str):
        """Save a checkpoint of the current training state.

        This method should be implemented by subclasses to save the current state
        of the training process, allowing it to be resumed later.

        Args:
            checkpoint_name: A name identifier for this checkpoint.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def cancel(self):
        """Cancel the reinforcement learning job.

        This method should be implemented by subclasses to cancel the ongoing
        training process and clean up resources.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def status(self):
        """Get the current status of the reinforcement learning job.

        This method should be implemented by subclasses to return the current
        status of the training job.

        Returns:
            The current status of the training job. The exact type depends on
            the provider implementation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class Provider:
    """Base class for language model providers.

    A provider is responsible for managing language model instances, including
    launching, killing, and fine-tuning models. Subclasses should implement
    provider-specific logic for these operations.

    Attributes:
        finetunable: Whether this provider supports fine-tuning.
        reinforceable: Whether this provider supports reinforcement learning.
        TrainingJob: The class to use for training jobs (subclass of `TrainingJob`).
        ReinforceJob: The class to use for reinforcement learning jobs (subclass of `ReinforceJob`).

    Example:
        ```python
        from dspy.clients.provider import Provider

        class MyProvider(Provider):
            def __init__(self):
                super().__init__()
                self.finetunable = True

            @staticmethod
            def is_provider_model(model: str) -> bool:
                return model.startswith("myprovider/")

            @staticmethod
            def finetune(job, model, train_data, train_data_format, train_kwargs=None):
                # Implement fine-tuning logic
                return "fine_tuned_model_id"
        ```
    """

    def __init__(self):
        self.finetunable = False
        self.reinforceable = False
        self.TrainingJob = TrainingJob
        self.ReinforceJob = ReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        """Check if a model identifier is supported by this provider.

        Subclasses should override this method to check whether a model is
        supported if they want to have the model provider auto-discovered.

        Args:
            model: The model identifier to check (e.g., "openai/gpt-4").

        Returns:
            True if the model is supported by this provider, False otherwise.
        """
        return False

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        """Launch a language model instance.

        This method should be implemented by subclasses to start a language model
        instance. Note that this method might be called even if there is already
        a launched LM, so implementations should be resilient to such cases.

        Args:
            lm: The language model instance to launch.
            launch_kwargs: Additional keyword arguments for launching the model.
                The `lm.launch_kwargs` dictionary will contain the necessary
                information for the provider to launch the LM.
        """
        pass

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        """Kill a running language model instance.

        This method should be implemented by subclasses to stop a running language
        model instance. Note that this method might be called even if there is no
        launched LM to kill, so implementations should be resilient to such cases.

        Args:
            lm: The language model instance to kill.
            launch_kwargs: Additional keyword arguments for killing the model.
                The `lm.launch_kwargs` dictionary will contain the necessary
                information for the provider to kill the LM. This argument is named
                `launch_kwargs` (not `kill_kwargs`) because it contains the same
                information used for launching.
        """
        pass

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Fine-tune a language model with the provided training data.

        This method should be implemented by subclasses to perform fine-tuning
        of a language model using the provided training data.

        Args:
            job: The training job instance to use for tracking the fine-tuning process.
            model: The model identifier to fine-tune.
            train_data: A list of training examples, each represented as a dictionary.
            train_data_format: The format of the training data. Can be a `TrainDataFormat`
                enum value or a string identifier.
            train_kwargs: Additional keyword arguments for fine-tuning configuration.

        Returns:
            The identifier of the fine-tuned model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
