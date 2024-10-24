# ChatModuleClient

## Prerequisites

1. Install the required packages using the following commands:
   
   ```shell
   pip install --no-deps --pre --force-reinstall mlc-ai-nightly-cu118 mlc-chat-nightly-cu118 -f https://mlc.ai/wheels
   pip install transformers
   git lfs install
   ```
   
   Adjust the pip wheels according to your OS/platform by referring to the provided commands in [MLC packages](https://mlc.ai/package/).

## Running MLC Llama-2 models

1. Create a directory for prebuilt models:

   ```shell
   mkdir -p dist/prebuilt
   ```
   
2. Clone the necessary libraries from the repository:

   ```shell
   git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib
   cd dist/prebuilt
   ```
   
3. Choose a Llama-2 model from [MLC LLMs](https://huggingface.co/mlc-ai) and clone the model repository:

   ```shell
   git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
   ```

4. Initialize the `ChatModuleClient` within your program with the desired parameters. Here's an example call:

   ```python
   llama = dspy.ChatModuleClient(model='dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1', model_path='dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so')
   ```
Please refer to the [official MLC repository](https://github.com/mlc-ai/mlc-llm) for more detailed information and [documentation](https://mlc.ai/mlc-llm/docs/get_started/try_out.html).
