from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from copy import deepcopy
from enum import Enum
from typing import Optional, Union, List
import ujson


class LM(ABC):
    """Abstract class for language models."""

    def __init__(self, model):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "default"
        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        import dspy

        if dspy.settings.experimental:
            return "\n\n" + "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end
        else:
            return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0, color_format: bool = True):
        """Prints the last n prompts and their completions.

        TODO: print the valid choice that contains filled output field instead of the first.
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                if provider in (
                    "clarifai",
                    "cloudflare",
                    "google",
                    "groq",
                    "Bedrock",
                    "Sagemaker",
                    "premai",
                    "tensorrt_llm",
                ):
                    printed.append((prompt, x["response"]))
                elif provider == "anthropic":
                    blocks = [
                        {"text": block.text}
                        for block in x["response"].content
                        if block.type == "text"
                    ]
                    printed.append((prompt, blocks))
                elif provider == "cohere":
                    printed.append((prompt, x["response"].text))
                elif provider == "mistral":
                    printed.append((prompt, x["response"].choices))
                elif provider == "ibm":
                    printed.append((prompt, x))
                elif provider == "you.com":
                    printed.append((prompt, x["response"]["answer"]))
                else:
                    printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        printing_value = ""
        for idx, (prompt, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n"
            printing_value += prompt

            text = ""
            if provider in (
                "cohere",
                "Bedrock",
                "Sagemaker",
                "clarifai",
                "claude",
                "ibm",
                "premai",
                "you.com",
                "tensorrt_llm",
            ):
                text = choices
            elif provider == "openai" or provider == "ollama" or provider == "llama":
                text = " " + self._get_choice_text(choices[0]).strip()
            elif provider == "groq":
                text = " " + choices
            elif provider == "google":
                text = choices[0].parts[0].text
            elif provider == "mistral":
                text = choices[0].message.content
            elif provider == "cloudflare":
                text = choices[0]
            else:
                text = choices[0]["text"]
            printing_value += self.print_green(text, end="") if color_format else text

            if len(choices) > 1 and isinstance(choices, list):
                choices_text = f" \t (and {len(choices)-1} other completions)"
                printing_value += self.print_red(
                   choices_text, end="",
                ) if color_format else choices_text

            printing_value += "\n\n\n"

        print(printing_value)
        return printing_value

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)


#-------------------------------------------------------------------------------
#    Classes for finetuning LMs
#-------------------------------------------------------------------------------

class TrainingMethod(str, Enum):
    """Enum class for training methods.
    
    When comparing enums, Python checks for object IDs, which means that the
    enums can't be compared directly. Subclassing the Enum class along with the
    str class allows for direct comparison of the enums.
    """
    SFT = "SFT"
    Preference = "Preference"


"""Dictionary mapping training methods to the data keys they require."""
TRAINING_METHOD_TO_DATA_KEYS = {
    TrainingMethod.SFT: ["prompt", "completion"],
    TrainingMethod.Preference: ["prompt", "chosen", "rejected"],
}

class TrainableLM(LM, ABC):
    """Base class for trainable LMs."""
    SUPPORTED_TRAINING_METHODS: Optional[List[TrainingMethod]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.SUPPORTED_TRAINING_METHODS is None:
            err_msg = f"{cls.__name__} must define 'SUPPORTED_TRAINING_METHODS'"
            raise NotImplementedError(err_msg)

    @abstractmethod
    def start_training(
        self,
        future: Future['TrainableLM'],
        method: TrainingMethod,
        train_path: str,
        **kwargs):
        """Start the training process asynchronously.
        
        This method should be implemented by the subclasses of TrainableLM to
        start the training process asynchronously.

        Args:
            future: A Future object that will hold the fine-tuned model once
                the training is complete.
            method: The training method to use.
            train_path: The path to the training data, which should be in the
                format required by the training method. The format for the
                selected training method is verified using
                self.verify_training_method_data_format(...) method.
            **kwargs: Additional arguments to be used for training.
        """
        lname = self.__class__.__name__
        err_msg = f"{lname} does not implement the 'start_training' method."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def stop_training(self) -> Exception:
        """Stop the any training process related to this instance.
        
        This method should be implemented by the subclasses of TrainableLM to
        stop any training process related to this instance.

        Raises:
            Exception: If the training process cannot be stopped or there is
                not training process to stop.
        """
        lname = self.__class__.__name__
        err_msg = f"{lname} does not implement the 'stop_training' method."
        raise NotImplementedError(err_msg)

    def get_supported_training_methods(self) -> List[TrainingMethod]:
        """Return the supported training methods for this class.
        
        This method is that can be called directly on the TrainableLM class to
        obtain the supported training methods for this class. The supported
        training methods are defined by the "SUPPORTED_TRAINING_METHODS" class
        variable of the subclass extending the TrainableLM class.

        Returns:
            List[TrainingMethod]: A list of the supported training methods.
        """
        return self.SUPPORTED_TRAINING_METHODS

    def verify_data_format_for_training_method(
        self,
        method: TrainingMethod,
        data_path: str
) -> Optional[AssertionError]:
        """Verify that the data at the given path is in the expected format.
        
        This method can be called directly on the LM instance to verify that the
        data at the given path is in the expected format for the given training
        method. If the data is not in the expected format, an AssertionError is
        raised. The expected format is always a list of dictionaries, with
        mandatory keys that are specific to the training method as specified
        below.
        - TrainingMethod.SFT: ["prompt", "completion"]
        - TrainingMethod.Preference: ["prompt", "chosen", "rejected"]

        This is to say that, for example, the data for the SFT training method
        should be a list of dictionaries, where each dictionary has the keys
        "prompt" and "completion". If any dictionary is missing one of these
        keys, an AssertionError is raised.

        Args:
            method: The training method to use.
            data_path: The path to the training data, which should be in the
                format required by the training method. The format for the
                selected training method is verified using this function.

        Returns:
            Optional[AssertionError]: An AssertionError if the data is not in
                the expected format, otherwise None.
        """
        expected_keys = TRAINING_METHOD_TO_DATA_KEYS[method]
        data = ujson.load(open(data_path))
        for ind, data_dict in enumerate(data):
            err_msg = f"The datapoint at index {ind} is missing the keys required for {method} training."
            err_msg = f"\n    Expected: {expected_keys}"
            err_msg = f"\n    Found: {data_dict.keys()}"
            assert all([key in data_dict for key in expected_keys]), err_msg

    def get_finetune(
            self,
            method: TrainingMethod, 
            train_path: str,
            **kwargs
        ) -> Union[Future['TrainableLM'], Exception]:
        """Return a future object that will hold the fine-tuned model.

        The user facing method of the TrainableLM models that starts
        asynchronous training and returns a future object that will hold the
        fine-tuned model once the training is complete.

        Args:
            method: The training method to use.
            train_path: The path to the training data, which should be in the
                format required by the training method. The format for the
                selected training method is verified using
                self.verify_training_method_data_format(...) method.
            **kwargs: Additional arguments that will be passed to the
                self.start_training(...) method. These arguments are specific to
                the particular subclass extending the TrainableLM class. For a
                description of what these should be for a particular subclass,
                refer to the documentation of the "start_training" method of
                that subclass.
    
        Returns:
            Future[TrainableLM]: A Future object that will hold the
                fine-tuned model. This future object can be polled repetitvely
                until it completes using the "done()" method. The results can
                then be obtained using the "result()" method. Shared below is an
                example. If the desired behavior is waiting until the training
                is complete, the "result()" method can be called directly, which
                will block until the training is complete. For more information
                on how to use Future objects, refer to the Python documentation
                at https://docs.python.org/3/library/concurrent.futures.html.

                ```
                import time
                ...

                future_lm = lm.get_finetune(...)

                while not future_lm.done():
                    time.sleep(60)
                
                lm = future_lm.result()
                ```
        """
        # TODO: Can we remove this circular import?
        import dspy
        err_msg = "The fine-tuning feature is experimental!"
        err_msg += " To use it, set dspy.settings.experimental = True."
        assert dspy.settings.experimental, err_msg

        # Input verification
        assert method in self.SUPPORTED_TRAINING_METHODS
        self.verify_data_format_for_training_method(
            method=method,
            data_path=train_path
        )

        # Create new model that will eventually be obtained through
        # future.result() when its "start_training" method completes its
        # execution
        future: Future['TrainableLM'] = Future()
        new_lm = deepcopy(self)

        # Capture the current instance in the closure and start the training
        # process asynchronously
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(
            new_lm.start_training,
            future,
            method=method,
            train_path=train_path,
            **kwargs
        )
        executor.shutdown(wait=False)

        return future
