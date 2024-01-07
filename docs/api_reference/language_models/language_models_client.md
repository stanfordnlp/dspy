# Language Model Clients


## Quickstart

```python
import dspy

lm = dspy.OpenAI(model='gpt-3.5-turbo')

prompt = "Translate the following English text to Spanish: 'Hi, how are you?'"
completions = lm(prompt, n=5, return_sorted=False)
for i, completion in enumerate(completions):
    print(f"Completion {i+1}: {completion}")
```

### Hosted

```{toctree}
---
maxdepth: 1
---
OpenAI <openai.md>
Cohere <cohere.md>
Anyscale <anyscale.md>
```

### Local

```{toctree}
---
maxdepth: 1
---

HFModel <HFModel.md>
TGI <HFTGI.md>
VLLM <vLLM.md>
LiteLLM <liteLLM.md>
``` 
