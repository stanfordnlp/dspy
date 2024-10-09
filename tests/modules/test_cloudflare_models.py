"""Tests for Cloudflare models.
Note: Requires configuration of your Cloudflare account_id and api_key.
"""

import dspy

models = {
    "@cf/qwen/qwen1.5-0.5b-chat": "https://huggingface.co/qwen/qwen1.5-0.5b-chat",
    "@hf/meta-llama/meta-llama-3-8b-instruct": "https://llama.meta.com",
    "@hf/nexusflow/starling-lm-7b-beta": "https://huggingface.co/Nexusflow/Starling-LM-7B-beta",
    "@cf/meta/llama-3-8b-instruct": "https://llama.meta.com",
    "@hf/thebloke/neural-chat-7b-v3-1-awq": "",
    "@cf/meta/llama-2-7b-chat-fp16": "https://ai.meta.com/llama/",
    "@cf/mistral/mistral-7b-instruct-v0.1": "https://mistral.ai/news/announcing-mistral-7b/",
    "@cf/tinyllama/tinyllama-1.1b-chat-v1.0": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "@hf/mistral/mistral-7b-instruct-v0.2": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    "@cf/fblgit/una-cybertron-7b-v2-bf16": "",
    "@hf/thebloke/codellama-7b-instruct-awq": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-AWQ",
    "@cf/thebloke/discolm-german-7b-v1-awq": "https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-AWQ",
    "@cf/meta/llama-2-7b-chat-int8": "https://ai.meta.com/llama/",
    "@hf/thebloke/mistral-7b-instruct-v0.1-awq": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    "@hf/thebloke/openchat_3.5-awq": "",
    "@cf/qwen/qwen1.5-7b-chat-awq": "https://huggingface.co/qwen/qwen1.5-7b-chat-awq",
    "@hf/thebloke/llama-2-13b-chat-awq": "https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ",
    "@hf/thebloke/deepseek-coder-6.7b-base-awq": "",
    "@hf/thebloke/openhermes-2.5-mistral-7b-awq": "",
    "@hf/thebloke/deepseek-coder-6.7b-instruct-awq": "",
    "@cf/deepseek-ai/deepseek-math-7b-instruct": "https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct",
    "@cf/tiiuae/falcon-7b-instruct": "https://huggingface.co/tiiuae/falcon-7b-instruct",
    "@hf/nousresearch/hermes-2-pro-mistral-7b": "https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B",
    "@hf/thebloke/zephyr-7b-beta-awq": "https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ",
    "@cf/qwen/qwen1.5-1.8b-chat": "https://huggingface.co/qwen/qwen1.5-1.8b-chat",
    "@cf/defog/sqlcoder-7b-2": "https://huggingface.co/defog/sqlcoder-7b-2",
    "@cf/microsoft/phi-2": "https://huggingface.co/microsoft/phi-2",
    "@hf/google/gemma-7b-it": "https://ai.google.dev/gemma/docs",
}


def get_lm(name: str): # -> dspy.LM:
    return dspy.CloudflareAI(model=name)


def run_tests():
    """Test the providers and models"""
    # Configure your AWS credentials with the AWS CLI before running this script
    models

    predict_func = dspy.Predict("question -> answer")
    for model_name in models.keys():
        lm = get_lm(model_name)
        with dspy.context(lm=lm):
            question = "What is the capital of France?"
            answer = predict_func(question=question).answer
            print(f"Question: {question}\nAnswer: {answer}")
            print("---------------------------------")
            lm.inspect_history()
            print("---------------------------------\n")


if __name__ == "__main__":
    run_tests()
