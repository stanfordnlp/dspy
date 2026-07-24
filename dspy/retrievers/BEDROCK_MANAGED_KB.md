# Bedrock Managed Knowledge Base Support

## Overview
Adds a DSPy retriever module that queries Amazon Bedrock Knowledge Bases for managed semantic search.

## Usage
```python
import dspy
from dspy.retrievers import BedrockKnowledgeBaseRetriever

retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="YOUR_KB_ID", k=5)
results = retriever("How do I configure IAM roles?")
# Returns list of dspy.Prediction with passage and score fields
```

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_ID | Bedrock Knowledge Base ID | None |
| AWS_REGION | AWS region for the KB | us-east-1 |
| AWS_PROFILE | AWS credentials profile | None |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |
| MAX_RESULTS | Maximum retrieval results | 5 |

## Features
- Managed search (no vector store needed)
- Agentic retrieval with query decomposition + reranking
- Automatic fallback to plain Retrieve if agentic fails
- Multi-source support (S3, Web, Confluence, SharePoint)
- Returns DSPy-native Prediction objects

## SDK Requirements
- boto3 >= 1.43
- dspy >= 2.0

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieveStream"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

## References
- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)
