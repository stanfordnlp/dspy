# """Tests for AWS models.
# Note: Requires configuration of your AWS credentials with the AWS CLI and creating sagemaker endpoints.
# TODO: Create mock fixtures for pytest to remove the need for AWS credentials and endpoints.
# """

# import dsp
# import dspy


# def get_lm(lm_provider: str, model_path: str, **kwargs) -> dsp.modules.lm.LM:
#     """get the language model"""
#     # extract model vendor and name from model name
#     # Model path format is <MODEL_VENDOR>/<MODEL_NAME_OR_ENDPOINT>
#     model_vendor = model_path.split("/")[0]
#     model_name = model_path.split("/")[1]

#     if lm_provider == "Bedrock":
#         bedrock = dspy.Bedrock(region_name="us-west-2")
#         if model_vendor == "mistral":
#             return dspy.AWSMistral(bedrock, model_name, **kwargs)
#         elif model_vendor == "anthropic":
#             return dspy.AWSAnthropic(bedrock, model_name, **kwargs)
#         elif model_vendor == "meta":
#             return dspy.AWSMeta(bedrock, model_name, **kwargs)
#         else:
#             raise ValueError(
#                 "Model vendor missing or unsupported: Model path format is <MODEL_VENDOR>/<MODEL_NAME_OR_ENDPOINT>"
#             )
#     elif lm_provider == "Sagemaker":
#         sagemaker = dspy.Sagemaker(region_name="us-west-2")
#         if model_vendor == "mistral":
#             return dspy.AWSMistral(sagemaker, model_name, **kwargs)
#         elif model_vendor == "meta":
#             return dspy.AWSMeta(sagemaker, model_name, **kwargs)
#         else:
#             raise ValueError(
#                 "Model vendor missing or unsupported: Model path format is <MODEL_VENDOR>/<MODEL_NAME_OR_ENDPOINT>"
#             )
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")


# def run_tests():
#     """Test the providers and models"""
#     # Configure your AWS credentials with the AWS CLI before running this script
#     provider_model_tuples = [
#         ("Bedrock", "mistral/mistral.mixtral-8x7b-instruct-v0:1"),
#         ("Bedrock", "anthropic/anthropic.claude-3-haiku-20240307-v1:0"),
#         ("Bedrock", "anthropic/anthropic.claude-3-sonnet-20240229-v1:0"),
#         ("Bedrock", "meta/meta.llama2-70b-chat-v1"),
#         ("Bedrock", "meta/meta.llama3-8b-instruct-v1:0"),
#         ("Bedrock", "meta/meta.llama3-70b-instruct-v1:0"),
#         # ('Sagemaker', 'mistral/<YOUR_ENDPOINT_NAME>'),  # REPLACE YOUR_ENDPOINT_NAME with your sagemaker endpoint
#     ]

#     predict_func = dspy.Predict("question -> answer")
#     for provider, model_path in provider_model_tuples:
#         print(f"Provider: {provider}, Model: {model_path}")
#         lm = get_lm(provider, model_path)
#         with dspy.context(lm=lm):
#             question = "What is the capital of France?"
#             answer = predict_func(question=question).answer
#             print(f"Question: {question}\nAnswer: {answer}")
#             print("---------------------------------")
#             lm.inspect_history()
#             print("---------------------------------\n")


# if __name__ == "__main__":
#     run_tests()
