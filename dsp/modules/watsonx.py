from typing import Any

from dsp.modules.lm import LM

ibm_watsonx_ai_api_error = False

try:
    import ibm_watsonx_ai  # noqa: F401
    from ibm_watsonx_ai.foundation_models import Model  # type: ignore

except ImportError:
    ibm_watsonx_ai_api_error = Exception


class Watsonx(LM):
    """Wrapper around Watsonx AI's API.

    The constructor initializes the base class LM to support prompting requests to Watsonx models.
    This requires the following parameters:
    Args:
        model (str): the type of model to use from IBM Watsonx AI.
        credentials ([dict]): credentials to Watson Machine Learning instance.
        project_id (str): ID of the Watson Studio project.
        **kwargs: Additional arguments to pass to the API provider. This is initialized with default values for relevant
            text generation parameters needed for communicating with Watsonx API, such as:
                - decoding_method
                - max_new_tokens
                - min_new_tokens
                - stop_sequences
                - repetition_penalty
    """

    def __init__(self, model, credentials, project_id, **kwargs):
        """Parameters

        model : str
            Which pre-trained model from Watsonx.ai to use?
            Choices are [
                `mistralai/mixtral-8x7b-instruct-v01`,
                `ibm/granite-13b-instruct-v2`,
                `meta-llama/llama-3-70b-instruct`]
        credentials : [dict]
            Credentials to Watson Machine Learning instance.
        project_id : str
            ID of the Watson Studio project.
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        self.model = model
        self.credentials = credentials
        self.project_id = project_id
        self.provider = "ibm"
        self.model_type = "instruct"
        self.kwargs = {
            "temperature": 0,
            "decoding_method": "greedy",
            "max_new_tokens": 150,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1,
            "num_generations": 1,
            **kwargs,
        }

        self.client = Model(
            model_id=self.model,
            params=self.kwargs,
            credentials=self.credentials,
            project_id=self.project_id,
        )

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> Any:
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}

        response = self.client.generate(prompt, params={**kwargs})

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt: str, **kwargs) -> Any:
        # Handles the specific prompting for each supported model and the retrieval of completions from IBM Watsonx AI

        if self.model == "mistralai/mixtral-8x7b-instruct-v01":
            prompt = "<s>[INST]" + prompt + "</INST>"
        elif self.model == "meta-llama/llama-3-70b-instruct":
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                + prompt
                + "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            )

        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Watsonx.

        Args:
            prompt (str): prompt to send to Watsonx
            only_completed (bool, optional): return only completed responses and ignores completion due to length.
            Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities.
            Defaults to False.
            **kwargs: Additional arguments to pass

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
        if only_completed is False:
            raise ValueError("only_completed is True for now")

        if return_sorted:
            raise ValueError("return_sorted is False for now")

        response = self.request(prompt, **kwargs)

        return [result["generated_text"] for result in response["results"]]
