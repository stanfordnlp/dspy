# Debugging DSPy Programs

## Common Issues and Solutions

### Context Length Errors

If you're encountering "context too long" errors in DSPy, this typically happens when using DSPy optimizers that include demonstrations within your prompt, exceeding your current context window. Here are some solutions:

1. **Reduce demonstration parameters**:
   - Lower `max_bootstrapped_demos` and `max_labeled_demos` in your optimizer settings
   - Reduce the number of retrieved passages/documents/embeddings

2. **Increase token limit**:
   ```python
   lm = dspy.OpenAI(model="gpt-4", max_tokens=2048)  # Adjust max_tokens as needed
   ```

### Timeouts and Backoff Errors

When dealing with timeout or backoff errors:

1. **Check provider status**:
   - Verify your LM/RM provider status
   - Ensure you have sufficient rate limits

2. **Adjust thread count**:
   - Reduce the number of concurrent threads to prevent server overload

3. **Configure backoff settings**:
   ```python
   # Global configuration
   dspy.settings.configure(backoff_time=...)

   # Provider-specific configuration
   with dspy.context(backoff_time=...):
       dspy.OpenAI(...)
   ```

### Caching Issues

To manage caching in DSPy:

1. **Disable cache**:
   ```python
   dspy.LM('openai/gpt-4', cache=False)
   ```

2. **Set cache directory**:
   ```python
   os.environ["DSP_NOTEBOOK_CACHE"] = os.path.join(os.getcwd(), ".dspy_cache")
   ```
