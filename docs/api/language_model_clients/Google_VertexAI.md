# GoogleVertexAI Usage Guide

This guide provides instructions on how to use the `GoogleVertexAI` class to interact with Google Vertex AI's API for text and code generation.

## Requirements

- Python 3.10 or higher.
- The `vertexai` package installed, which can be installed via pip.
- A Google Cloud account and a configured project with access to Vertex AI.

## Installation

Ensure you have installed the `vertexai` package along with other necessary dependencies:

```bash
pip install dspy-ai[google-vertex-ai]
```

## Configuration

Before using the `GoogleVertexAI` class, you need to set up access to Google Cloud:

1. Create a project in Google Cloud Platform (GCP).
2. Enable the Vertex AI API for your project.
3. Create authentication credentials and save them in a JSON file.

## Usage

Here's an example of how to instantiate the `GoogleVertexAI` class and send a text generation request:

```python
from dsp.modules import GoogleVertexAI  # Import the GoogleVertexAI class

# Initialize the class with the model name and parameters for Vertex AI
vertex_ai = GoogleVertexAI(
    model_name="text-bison@002",
    project="your-google-cloud-project-id",
    location="us-central1",
    credentials="path-to-your-service-account-file.json"
)
```

## Customizing Requests

You can customize requests by passing additional parameters such as `temperature`, `max_output_tokens`, and others supported by the Vertex AI API. This allows you to control the behavior of the text generation.

## Important Notes

- Make sure you have correctly set up access to Google Cloud to avoid authentication issues.
- Be aware of the quotas and limits of the Vertex AI API to prevent unexpected interruptions in service.

With this guide, you're ready to use `GoogleVertexAI` for interacting with Google Vertex AI's text and code generation services.
