# Setting up DSPy

Use pip >= 26.2.2 or uv >= 0.13.4. If you have an older installer, follow the
[upgrade instructions below](#upgrading-existing-environments) before installing packages.

Install DSPy with `pip install dspy` or run `uv add dspy` to add DSPy to a virtual environment. DSPy works in Python 3.10+ environments.

## Upgrading existing environments

The [October 2026 emSign certificate incident](https://discuss.python.org/t/109576)
requires updated installers and CA bundles. Updating certifi alone does not repair
pip's or uv's bundled certificates.

In each Python 3.10+ environment, bootstrap pip using this SHA-256-verified wheel,
then update uv and certifi with the repaired installer:

```bash
python -m pip --disable-pip-version-check install --no-deps --upgrade \
  'https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
python -m pip install --upgrade 'uv>=0.13.4' 'certifi>=2026.10.19'
```

Run this after creating a venv with `python -m venv` too: Python's `ensurepip` may
still seed an old pip. For pip-free uv environments, upgrade the uv executable
from another repaired environment, then run `uv pip install 'certifi>=2026.10.19'`.
Check `uv --version` to ensure your shell does not select an older standalone copy.
Update OS CA packages and rebuild container images as well. Existing environments
and cached pre-commit hooks are not repaired by changing a lockfile alone.

## Connecting to a language model

DSPy connects to language models with the `dspy.LM` class. To set up a language model, we provide a `"provider/model"` format string and an API key:

```py
import dspy

# Pass the key explicitly...
lm = dspy.LM("openai/gpt-5-nano", api_key="YOUR_OPENAI_API_KEY")

dspy.configure(lm=lm)
```

Behind the scenes, DSPy uses the [LiteLLM](https://docs.litellm.ai/docs/#litellm-python-sdk) library to normalize inference providers into a single format. This allows you to provide a LiteLLM model string and connect to nearly any model and its provider. [Click here to search for the full list](https://models.litellm.ai/).

For this tutorial, we could replace the `"openai/gpt-5-nano"` model string and our OpenAI API key with options from Anthropic, Google, OpenRouter, and more. All the examples below will work with any LiteLLM model string, without code changes.

Once we have an `LM`, calling `dspy.configure(lm=lm)` sets our `LM` as the default provider for every DSPy program in the process. This sets our `LM` globally, but we can selectively override this with [`dspy.context`](../diving-deeper/settings-and-context.md) when more granular control is needed.

Let’s ensure everything works by manually calling the model:

```py
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

print(lm(messages = messages))

```

Which returns:

```
['The capital of France is Paris.']
```

With an `LM` configured, we can proceed to writing our first program.

---

**Next:** [Your first program →](first-program.md)
