# dspy.LM

<!-- START_API_REF -->
::: dspy.LM
    handler: python
    options:
        members:
            - __call__
            - acall
            - copy
            - dump_state
            - finetune
            - infer_provider
            - inspect_history
            - kill
            - launch
            - load_state
            - reinforce
            - update_history
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

## Choosing an engine

`engine="auto"` prefers native lm15 and may select the optional LiteLLM
compatibility engine for unsupported routes, settings or representations before
provider I/O. `engine="lm15"` requires native; `engine="litellm"` explicitly
selects compatibility. Provider failures never trigger a backend switch.
Native imports and calls do not require LiteLLM. Compatibility use requires
`pip install "dspy[litellm]"`; without it, an actionable error names that command.
`model_type="text"` uses compatibility, not native lm15. All engines retain the
canonical `Request`/`Response` contract. See the
[migration guide](../../community/normalized-lm-api-migration.md) for limitations.

## Application-owned exception: text-only Bedrock Nova

Native Bedrock Nova/Converse is not supported by this DSPy release. The following
recipe is a bounded, application-owned exception, **not restored native support
or an accepted migration-gate exception**. A custom engine is not the only
integration mechanism: optional LiteLLM compatibility also remains available.
This recipe does not verify built-in Nova execution through that engine. It uses the public
[`complete(Request) -> Response` engine contract](../../tutorials/custom_lm_engines/index.md#what-a-custom-engine-owns)
and LiteLLM installed through DSPy's optional extra, pinned to the version tested here:

```shell
pip install "dspy[litellm]" "litellm==1.96.0"
```

This example targets **`amazon.nova-lite-v1:0` in `us-east-1`**, using AWS
credentials configured for your application. It forwards a string system prompt
and all user/assistant text blocks, including demonstrations. Only `max_tokens`
and `temperature` (0–1) are supported. Tools, media, developer/tool messages,
continuation state, structured outputs, reasoning, provider-side caching and
other non-default generation settings are refused before the SDK call.

```python
import asyncio
from dataclasses import replace

import dspy
import litellm
from openai import APIError  # LiteLLM's SDK errors inherit this public type.
from dspy import lm15

MODEL = "bedrock/amazon.nova-lite-v1:0"
REGION = "us-east-1"
litellm.num_retries = 0  # Application-owned LiteLLM setting; DSPy owns retries.


class NovaText:
    supported_params = {"max_tokens", "temperature"}

    def complete(self, request: lm15.Request) -> lm15.Response:
        config = request.config
        if request.model != MODEL or request.tools or not isinstance(request.system, (str, type(None))):
            raise lm15.UnsupportedFeatureError("Nova recipe requires its fixed model, no tools, and a string system prompt")
        if replace(config, max_tokens=None, temperature=None) != lm15.Config():
            raise lm15.UnsupportedFeatureError("Nova recipe supports only max_tokens and temperature")
        if config.temperature is not None and config.temperature > 1:
            raise lm15.UnsupportedFeatureError("Nova recipe requires temperature between 0 and 1")
        messages = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if (message.role not in {"user", "assistant"} or message.text is None
                    or message.continuation or any(part.continuation for part in message.parts)):
                raise lm15.UnsupportedFeatureError("Nova recipe accepts only text messages without continuation state")
            messages.append({"role": message.role, "content": [
                {"type": "text", "text": part.text} for part in message.parts
            ]})
        options = {key: getattr(config, key) for key in self.supported_params if getattr(config, key) is not None}
        try:
            raw = litellm.completion(
                model=MODEL, messages=messages, aws_region_name=REGION,
                timeout=30, num_retries=0, caching=False, drop_params=False, **options,
            )
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            error = {400: lm15.InvalidRequestError, 401: lm15.AuthError,
                     403: lm15.AuthError, 429: lm15.RateLimitError}.get(status, lm15.ProviderError)
            if isinstance(exc, litellm.Timeout):
                error = lm15.TimeoutError
            elif isinstance(exc, litellm.APIConnectionError):
                # LiteLLM also uses this class for unclassified HTTP failures.
                error, status = lm15.ProviderError, None
            elif status is not None and status >= 500:
                error = lm15.ServerError
            raise error(str(exc), provider="bedrock", status=status) from exc
        response = lm15.response_from_openai_chat(raw.model_dump(), model=request.model)
        if response.message.text is None:
            raise lm15.UnsupportedFeatureError("Nova recipe received a non-text response")
        return response


class AsyncNovaText(NovaText):
    async def complete(self, request: lm15.Request) -> lm15.Response:
        return await asyncio.to_thread(super().complete, request)
```

Use the pair for ordinary synchronous and asynchronous prediction:

```python
lm = dspy.LM(MODEL, engine=NovaText(), async_engine=AsyncNovaText(),
             temperature=0, max_tokens=128, cache=True, num_retries=2)
dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))
program = dspy.Predict("question -> answer")
print(program(question="What is the capital of France?").answer)
print(asyncio.run(program.acall(question="What is the capital of Germany?")).answer)
```

In an existing event loop, use `await program.acall(...)` instead of
`asyncio.run(...)`. The async engine runs the same blocking SDK call in a worker
thread; cancelling the await does not stop an in-flight HTTP request. The SDK
timeout is 30 seconds. There is no streaming or engine serialization in this
recipe; reattach the engine after loading a program saved without its LM.

DSPy owns response caching and retry limits; LiteLLM caching and retries are
disabled here. Keep this application's LiteLLM retry setting at zero and do not
add SDK/router fallback or retry policies. Use an isolated DSPy cache for this
deployment; changing AWS account, region or endpoint does not itself partition
DSPy's response cache. The public lm15 response reader preserves reported usage,
finish reasons and provider data; this recipe makes no cost-accounting promise.

Retry classification is limited to what LiteLLM identifies: rate limits,
timeouts and server failures are retryable. Its generic `APIConnectionError`
can also represent an unclassified HTTP failure (observed with a local 403), so
this recipe reports it as a non-retryable `LMProviderError`, without claiming an
HTTP status. The SDK exception remains in the cause chain; connection failures
are deliberately not retried automatically either.

**Validation boundary:** the copied recipe was executed through `dspy.Predict`
with LiteLLM 1.96.0 against a disposable loopback Converse endpoint, with fake AWS
credentials and locally checked SigV4 signatures. Sync/async text output,
request model/body, local refusals, SDK failures, DSPy retries and cache hits were
checked. This is local protocol validation with canned responses, **not live AWS
validation**: AWS model availability, IAM permissions, regional support and real
Nova output quality remain unverified. It does not establish acceptance of this
exception by the migration gate.
