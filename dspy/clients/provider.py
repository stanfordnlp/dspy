"""Provider module for DSPy model fine-tuning and reinforcement learning.

This module provides base classes for implementing model providers that support
fine-tuning and reinforcement learning workflows.
"""

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
    for monitoring and managing model training operations. Subclasses should implement
    provider-specific training logic.
    
    Attributes:
        thread: The thread executing the training job, if applicable.
        model: The name or identifier of the model being trained.
        train_data: The training data used for fine-tuning.
        train_data_format: The format of the training data.
        train_kwargs: Additional keyword arguments for training configuration.
    
    Example:
        >>> job = TrainingJob(
        ...     model="gpt-4",
        ...     train_data=[{"prompt": "Hello", "response": "Hi there!"}],
        ...     train_data_format=TrainDataFormat.CHAT,
        ... )
        >>> job.status()  # Check job status
    """

    def __init__(
        self,
        thread: Thread | None = None,
        model: str | None = None,
        train_data: list[dict[str, Any]] | None = None,
        train_data_format: TrainDataFormat | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize a new TrainingJob.
        
        Args:
            thread: Optional thread executing the training job.
            model: The model identifier to fine-tune.
            train_data: List of training examples.
            train_data_format: Format specification for training data.
            train_kwargs: Additional training configuration parameters.
        """
        self.thread = thread
        self.model = model
        self.train_data = train_data
        self.train_data_format = train_data_format
        self.train_kwargs = train_kwargs or {}
        super().__init__()

    def cancel(self):
        """Cancel the training job.
        
        Subclasses should override this method to implement provider-specific
        cancellation logic, then call super().cancel() to properly cancel
        the underlying Future.
        """
        super().cancel()

    @abstractmethod
    def status(self):
        """Get the current status of the training job.
        
        Returns:
            Provider-specific status information about the training job.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


class ReinforceJob:
    """Manages reinforcement learning training sessions for language models.
    
    ReinforceJob provides an interface for iterative reinforcement learning,
    supporting checkpointing and incremental training steps.
    
    Attributes:
        lm: The language model being trained.
        train_kwargs: Additional training configuration parameters.
        checkpoints: Dictionary mapping checkpoint names to saved states.
        last_checkpoint: The most recently saved checkpoint name.
    
    Example:
        >>> from dspy import LM
        >>> lm = LM("openai/gpt-4")
        >>> job = ReinforceJob(lm=lm, train_kwargs={"learning_rate": 1e-5})
        >>> job.initialize()
        >>> job.step(train_data=[...])
        >>> job.save_checkpoint("epoch_1")
        >>> job.terminate()
    """

    def __init__(self, lm: "LM", train_kwargs: dict[str, Any] | None = None):
        """Initialize a new ReinforceJob.
        
        Args:
            lm: The language model instance to train.
            train_kwargs: Optional training configuration parameters.
        """
        self.lm = lm
        self.train_kwargs = train_kwargs or {}
        self.checkpoints = {}
        self.last_checkpoint = None

    @abstractmethod
    def initialize(self):
        """Initialize the reinforcement learning environment.
        
        This method should be called before any training steps. It sets up
        the necessary resources and configurations for training.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | str | None = None):
        """Execute a single training step.
        
        Args:
            train_data: List of training examples for this step.
            train_data_format: Optional format specification for the training data.
                Can be a TrainDataFormat enum value or a string identifier.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        """Terminate the reinforcement learning session.
        
        This method should clean up any resources allocated during training
        and finalize the training process.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str):
        """Save a checkpoint of the current training state.
        
        Args:
            checkpoint_name: A unique identifier for this checkpoint.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def cancel(self):
        """Cancel the reinforcement learning job.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def status(self):
        """Get the current status of the reinforcement learning job.
        
        Returns:
            Provider-specific status information.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


class Provider:
    """Base class for DSPy model providers.
    
    Provider defines the interface for integrating external model services
    with DSPy's fine-tuning and reinforcement learning capabilities.
    Subclasses should implement provider-specific logic for model management.
    
    Attributes:
        finetunable: Whether this provider supports fine-tuning.
        reinforceable: Whether this provider supports reinforcement learning.
        TrainingJob: The TrainingJob class used by this provider.
        ReinforceJob: The ReinforceJob class used by this provider.
    
    Example:
        >>> class MyProvider(Provider):
        ...     @staticmethod
        ...     def is_provider_model(model: str) -> bool:
        ...         return model.startswith("my-provider/")
    """

    def __init__(self):
        """Initialize the Provider with default settings."""
        self.finetunable = False
        self.reinforceable = False
        self.TrainingJob = TrainingJob
        self.ReinforceJob = ReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        """Check if a model identifier is supported by this provider.
        
        Subclasses should override this method to implement provider-specific
        model detection logic for auto-discovery functionality.
        
        Args:
            model: The model identifier to check.
            
        Returns:
            True if the model is supported by this provider, False otherwise.
        """
        return False

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        """Launch a language model instance.
        
        This method is called to initialize and start a language model.
        The implementation should be resilient to being called when
        a model is already launched.
        
        Args:
            lm: The language model to launch.
            launch_kwargs: Optional keyword arguments for launching.
        """
        pass

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        """Stop a running language model instance.
        
        This method is called to shut down a language model. The implementation
        should be resilient to being called when no model is running.
        
        Note:
            The launch_kwargs parameter contains the same configuration used
            during launch, enabling proper cleanup of resources.
        
        Args:
            lm: The language model to stop.
            launch_kwargs: Optional keyword arguments (same as used for launch).
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
        """Execute a fine-tuning job for the specified model.
        
        Args:
            job: The TrainingJob instance managing this operation.
            model: The model identifier to fine-tune.
            train_data: List of training examples.
            train_data_format: Format specification for the training data.
            train_kwargs: Additional training configuration parameters.
            
        Returns:
            The identifier of the fine-tuned model.
            
        Raises:
            NotImplementedError: If fine-tuning is not supported by this provider.
        """
        raise NotImplementedError
