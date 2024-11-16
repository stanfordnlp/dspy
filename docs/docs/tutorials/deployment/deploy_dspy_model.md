# Deploy DSPy Model

This guide demonstrates two popular approaches to deploy your DSPy models in production:

1. FastAPI - For simple, lightweight deployments with direct model access
2. MLflow - For production-grade deployments with model versioning and management

## Deploying with FastAPI

FastAPI offers a quick and straightforward way to serve your DSPy model as a REST API. This approach is ideal when you have direct access to your model code and need a lightweight deployment solution.

Before we get started, let's install the required libraries:

```bash
pip install fastapi uvicorn
```

And remember to set your OpenAI API key which is used by our example:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Here's a minimal example of a DSPy model:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
dspy_model = dspy.ChainOfThought("question -> answer")
```

Create a FastAPI application to serve this model:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Model API",
    description="A simple API serving a DSPy Chain of Thought model",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and Chain of Thought
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm, async_max_workers=4) # default is 8
dspy_model = dspy.asyncify(dspy.ChainOfThought("question -> answer"))

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_model(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

In the above code, we call `dspy.asyncify(dspy.ChainOfThought("question -> answer"))` to convert the dspy program to run
in async mode, which is required by high-throughput FastAPI deployments. This under the hood runs the dspy program in a
separate thread and await its result. By default, the limit of spawned threads is 8. Think of this like a worker pool
(like ThreadPoolExecutor). If you have 8 in-flight programs and call it once more, the 9th call will wait until one of the
8 returns.

You can configure the async capacity using the new `async_max_workers` setting, e.g. to set it to 16, which will be shared
by all async calls:

```python
dspy.settings.configure(async_max_workers=16)
```

> ℹ️ **Note:** Recommended practice is to set async_max_workers once at the entry point of your program. If you configure it multiple
> times, then the last value will be used as the global async max workers value for all async calls.

Save your code in a file, e.g., `fastapi_dspy.py`. Then you can run the app with:

```bash
uvicorn fastapi_dspy:app --reload
```

It will start a local server at `http://127.0.0.1:8000/`. You can test it with the python code below:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "What is the capital of France?"}
)
print(response.json())
```

You should see the response like below:

```json
{'status': 'success', 'data': {'reasoning': 'The capital of France is a well-known fact, commonly taught in geography classes and referenced in various contexts. Paris is recognized globally as the capital city, serving as the political, cultural, and economic center of the country.', 'answer': 'The capital of France is Paris.'}}
```

## Deploying with MLflow

We recommend deploying with MLflow if you are looking for a solution to package your DSPy model and deploy in an isolated environment.
MLflow is a popular platform for managing machine learning workflows, including model versioning, tracking, and deployment.

Let's first install the required libraries:

```bash
pip install mlflow>=2.18.0
```

Let's spin up the MLflow tracking server, where we will store our DSPy model. The command below will start a local server at
`http://127.0.0.1:5000/`.

```bash
mlflow ui
```

Then we can define the DSPy model and log it to the MLflow server. "log" is an overloaded term in MLflow, basically it means
we store the model information along with environment requirements in the MLflow server. See the code below:

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_model")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
dspy_model = dspy.ChainOfThought("question -> answer")

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_model,
        "dspy_model",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

We recommend you to set `task="llm/v1/chat"` so that the deployed model automatically takes input and generate output in
the same format as the OpenAI chat API, which is a common interface for LLM applications nowadays. Save the code above in
a file, e.g., `mlflow_dspy.py`, and run it.

After you logged the model, you can view the saved information in MLflow UI. Open `http://127.0.0.1:5000/` and select
the `deploy_dspy_model` experiment, then select the run your just created, under the `Artifacts` tab, you should see the
logged model information, similar to the following screenshot:

![MLflow UI](./dspy_mlflow_ui.png)

Grab your run id from UI (or the console print when you execute `mlflow_dspy.py`), now you can deploy the logged model
with the following command:

```bash
mlflow models serve -m runs:/{run_id}/model -p 6000
```

After the model is deployed, you can test it with the following command:

```bash
curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json"  --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

You should see the response like below:

```json
{"choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"reasoning\": \"The question asks for the sum of 2 and 2. To find the answer, we simply add the two numbers together: 2 + 2 = 4.\", \"answer\": \"4\"}"}, "finish_reason": "stop"}]}
```

For complete guide on how to deploy a DSPy model with MLflow, and how to customize the deployment, please refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

### Best Practices for MLflow Deployment

1. **Environment Management**: Always specify your Python dependencies in a `conda.yaml` or `requirements.txt` file.
2. **Model Versioning**: Use meaningful tags and descriptions for your model versions.
3. **Input Validation**: Define clear input schemas and examples.
4. **Monitoring**: Set up proper logging and monitoring for production deployments.

For production deployments, consider using MLflow with containerization:

```bash
mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-model"
docker run -p 6000:8080 dspy-model
```

For a complete guide on production deployment options and best practices, refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).
