# dspy errors

DSPy exposes structured exception classes for LM failures and adapter parsing failures. LM errors inherit from `dspy.LMError`, which inherits from `dspy.DSPyError` and carries metadata such as `model`, `provider`, `status`, `request_id`, and `retry_after` when available.

Use `dspy.LMError` to catch any LM call failure, or catch a concrete subclass when you need specific handling.

```python
try:
    result = program(question="...")
except dspy.ContextWindowExceededError as e:
    print(f"Prompt was too long for {e.model}")
except dspy.LMRateLimitError as e:
    print(f"Rate limited; retry after {e.retry_after} seconds")
except dspy.LMError as e:
    print(f"LM failed with code={e.code}, provider={e.provider}, request_id={e.request_id}")
```

Use `dspy.is_retryable_lm_error(error)` to classify LM failures that are generally safe to retry: rate limits, timeouts, server errors, and transport errors. DSPy's built-in `dspy.LM` delegates provider retries to LiteLLM, but callers can use this helper after retries are exhausted. Retryability is advisory: respect provider policy and `retry_after` when present.

```python
try:
    result = program(question="...")
except dspy.LMError as e:
    if dspy.is_retryable_lm_error(e):
        # Schedule another attempt later.
        raise
    raise
```

## API Reference

::: dspy.utils.exceptions
    handler: python
    options:
        members:
            - DSPyError
            - LMError
            - LMTransportError
            - LMConfigurationError
            - LMNotConfiguredError
            - LMUnsupportedFeatureError
            - LMProviderError
            - LMUnexpectedError
            - LMAuthError
            - LMBillingError
            - LMRateLimitError
            - LMInvalidRequestError
            - ContextWindowExceededError
            - LMUnsupportedModelError
            - LMTimeoutError
            - LMServerError
            - is_retryable_lm_error
            - AdapterParseError
        show_source: true
        show_root_heading: false
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
