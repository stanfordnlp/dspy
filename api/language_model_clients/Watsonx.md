---
sidebar_position: 12
---

# dspy.Watsonx

This guide provides instructions on how to use the `Watsonx` class to interact with  IBM Watsonx.ai API for text and code generation.

## Requirements

- Python 3.10 or higher.
- The `ibm-watsonx-ai` package installed, which can be installed via pip.
- An IBM Cloud account and a Watsonx configured project.

## Installation

Ensure you have installed the `ibm-watsonx-ai` package along with other necessary dependencies:

## Configuration

Before using the `Watsonx` class, you need to set up access to IBM Cloud:

1. Create an IBM Cloud account
2. Enable a Watsonx service from the catalog
3. Create a new project and associate a Watson Machine Learning service instance.
4. Create an IAM authentication credentials and save them in a JSON file.

## Usage

Here's an example of how to instantiate the `Watsonx` class and send a generation request:

```python
import dspy

''' Initialize the class with the model name and parameters for Watsonx.ai
    You can choose between many different models:
    * (Mistral) mistralai/mixtral-8x7b-instruct-v01
    * (Meta) meta-llama/llama-3-70b-instruct
    * (IBM) ibm/granite-13b-instruct-v2
    * and many others.
'''
watsonx=dspy.Watsonx(
    model='mistralai/mixtral-8x7b-instruct-v01',
    credentials={
        "apikey": "your-api-key",
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id="your-watsonx-project-id",
    max_new_tokens=500,
    max_tokens=1000
    )

dspy.settings.configure(lm=watsonx)
```

## Customizing Requests

You can customize requests by passing additional parameters such as `decoding_method`,`max_new_tokens`, `stop_sequences`, `repetition_penalty`, and others supported by the Watsonx.ai API. This allows you to control the behavior of the generation.
Refer to [`ibm-watsonx-ai library`](https://ibm.github.io/watsonx-ai-python-sdk/index.html) documentation.
