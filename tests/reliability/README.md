# DSPy Reliability Tests

This directory contains reliability tests for DSPy programs. The purpose of these tests is to verify that DSPy programs reliably produce expected outputs across multiple large language models (LLMs), regardless of model size or capability. These tests are designed to ensure that DSPy programs maintain robustness and accuracy across diverse LLM configurations.

### Overview

Each test in this directory executes a DSPy program using various LLMs. By running the same tests across different models, these tests help validate that DSPy programs handle a wide range of inputs effectively and produce reliable outputs, even in cases where the model might struggle with the input or task.

### Key Features

- **Diverse LLMs**: Each DSPy program is tested with multiple LLMs, ranging from smaller models to more advanced, high-performance models. This approach allows us to assess the consistency and generality of DSPy program outputs across different model capabilities.
- **Challenging and Adversarial Tests**: Some of the tests are intentionally challenging or adversarial, crafted to push the boundaries of DSPy. These challenging cases allow us to gauge the robustness of DSPy and identify areas for potential improvement.
- **Cross-Model Compatibility**: By testing with different LLMs, we aim to ensure that DSPy programs perform well across model types and configurations, reducing model-specific edge cases and enhancing program versatility.

### Running the Tests

- First, populate the configuration file `reliability_tests_conf.yaml` (located in this directory) with the necessary LiteLLM model/provider names and access credentials for 1. each LLM you want to test and 2. the LLM judge that you want to use for assessing the correctness of outputs in certain test cases. These should be placed in the `litellm_params` section for each model in the defined `model_list`. You can also use `litellm_params` to specify values for LLM hyperparameters like `temperature`. Any model that lacks configured `litellm_params` in the configuration file will be ignored during testing.

  The configuration must also specify a DSPy adapter to use when testing, e.g. `"chat"` (for `dspy.ChatAdapter`) or `"json"` (for `dspy.JSONAdapter`).

  An example of `reliability_tests_conf.yaml`:

      ```yaml
      adapter: chat
      model_list:
        # The model to use for judging the correctness of program
        # outputs throughout reliability test suites. We recommend using
        # a high quality model as the judge, such as OpenAI GPT-4o
        - model_name: "judge"
          litellm_params:
            model: "openai/gpt-4o"
            api_key: "<my_openai_api_key>"
        - model_name: "gpt-4o"
          litellm_params:
            model: "openai/gpt-4o"
            api_key: "<my_openai_api_key>"
        - model_name: "claude-3.5-sonnet"
          litellm_params:
            model: "anthropic/claude-3.5"
            api_key: "<my_anthropic_api_key>"

- Second, to run the tests, run the following command from this directory:

  ```bash
      pytest .
  ```

  This will execute all tests for the configured models and display detailed results for each model configuration. Tests are set up to mark expected failures for known challenging cases where a specific model might struggle, while actual (unexpected) DSPy reliability issues are flagged as failures (see below).

#### Running specific generated tests

You can run specific generated tests by using the `-k` flag with `pytest`. For example, to test the generated program located at `tests/reliability/complex_types/generated/test_nesting_1` against generated test input `input1.json`, you can run the following command from this directory:

```bash
pytest test_generated.py -k "test_nesting_1-input1"
```

### Test generation

You can generate test DSPy programs and test inputs from text descriptions using the `tests.reliability.generate` CLI, or the `tests.reliability.generate.generate_test_cases` API. For example, to generate a test classification program and 3 challenging test inputs in the `tests/reliability/classification/generated` directory, you can run the following command from the DSPy repository root directory:

```bash
python \
    -m tests.reliability.generate \
    -d tests/reliability/classification/generated/test_example \
    -p "Generate a program that performs a classification task involving objects with multiple properties. The task should be realistic" \
    -i "Based on the program description, generate a challenging example" \
    -n 3
```

The test program will be written to `tests/reliability/classification/generated/test_example/program.py`, and the test inputs will be written as JSON files to the `tests/reliability/classification/generated/test_exaple/inputs/` directory.

All generated tests should be located in directories with the structure `tests/reliability/<test_type>/generated/<test_name>`, where `<test_type>` is the type of test (e.g., `classification`, `complex_types`, `chat`, etc.), and `<test_name>` is a descriptive name for the test.

### Known Failing Models

Some tests may be expected to fail with certain models, especially in challenging cases. These known failures are logged but do not affect the overall test result. This setup allows us to keep track of model-specific limitations without obstructing general test outcomes. Models that are known to fail a particular test case are specified using the `@known_failing_models` decorator. For example:

```
@known_failing_models(["llama-3.2-3b-instruct"])
def test_program_with_complex_deeply_nested_output_structure():
    ...
```
