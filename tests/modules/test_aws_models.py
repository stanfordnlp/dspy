"""
Testing the AWS modules
"""
import dspy
from dspy import AWSAnthropic, AWSLlama2, AWSMistral, AWSModel, Bedrock, Sagemaker


class QASignature(dspy.Signature):
    """answer the question"""

    question = dspy.InputField()
    answer = dspy.OutputField()


def test_aws_models(lm: AWSModel):
    """Test the models on the given AWS provider"""

    predict_func = dspy.ChainOfThought(QASignature)
    with dspy.context(lm=lm):
        answer = predict_func(question="What is the capital of France")
        assert "paris" in str(answer).lower()
        print(answer)


if __name__ == "__main__":
    # NOTE: Configure your AWS credentials with the AWS CLI before running this test!

    bedrock = Bedrock(region_name="us-west-2")
    test_aws_models(AWSMistral(bedrock, model="mistral.mixtral-8x7b-instruct-v0:1"))
    test_aws_models(AWSMistral(bedrock, "mistral.mistral-7b-instruct-v0:2"))
    test_aws_models(AWSAnthropic(bedrock, "anthropic.claude-3-haiku-20240307-v1:0"))
    test_aws_models(AWSAnthropic(bedrock, "anthropic.claude-3-sonnet-20240229-v1:0"))
    # this is slower than molasses and generates irrelevant content after the answer!!
    # You may have to wait for 10-15min before it returns
    test_aws_models(AWSLlama2(bedrock, "meta.llama2-70b-chat-v1"))

    # NOTE: Configure your Sagemaker endpoints before running this test!
    sagemaker = Sagemaker(region_name="us-west-2")
    # NOTE: Replace model value below with your own endpoint name
    test_aws_models(AWSMistral(sagemaker, model="g5-48xlarge-mixtral-8x7b"))
