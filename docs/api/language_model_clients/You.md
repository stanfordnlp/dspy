---
sidebar_position: 13
---

# dspy.You
Wrapper around [You.com's conversational Smart and Research APIs](https://documentation.you.com/api-reference/).

Each API endpoint is designed to generate conversational
responses to a variety of query types, including inline citations
and web results when relevant.

Smart Mode:
- Quick, reliable answers for a variety of questions
- Cites the entire web page URL

Research Mode:
- In-depth answers with extensive citations for a variety of questions
- Cites the specific web page snippet relevant to the claim

For more information, check out the documentations at
https://documentation.you.com/api-reference/.


### Constructor
```python
You(
    mode: Literal["smart", "research"] = "smart",
    api_key: Optional[str] = None,
)
```

**Parameters:**
- `mode`: You.com conversational endpoints. Choose from "smart" or "research"
- `api_key`: You.com API key, if `YDC_API_KEY` is not set in the environment

### Usage
Obtain a You.com API key from https://api.you.com/.

Export this key to an environment variable `YDC_API_KEY`.

```python
import dspy

# The API key is inferred from the `YDC_API_KEY` environment variable
lm = dspy.You(mode="smart")
```
