# Adapters: how signatures become prompts

## Intent

An adapter is the layer between a `Signature` and the LM. It formats the signature’s instructions, fields, and demos into chat messages, sends the call, and parses the LM’s response back into typed Python values. Different adapters use different prompt shapes — chat markers, JSON, XML, or a two-stage extract — so the same Signature can run against models with widely different formatting strengths.

Read this when you want to know what the prompt looks like on the wire, why a typed field parses one way and not another, where the `[[ ## field_name ## ]]` markers come from, or which adapter to switch to when ChatAdapter is misbehaving.

## Design decisions

### 1. Adapters are pluggable

Same Signature, different prompt shape, no Signature changes. LMs vary a lot in what they prefer to read and produce: a model trained on instruction-following with markers loves ChatAdapter’s `[[ ## field ## ]]` format; a model with native structured-output mode does best with JSONAdapter; a reasoning model that formats unreliably wants TwoStepAdapter. Putting this decision behind one interface means Signatures, modules, and optimizers don’t have to know which family of LM they’re running against.

### 2. ChatAdapter is the default

Text-only, model-agnostic, uses `[[ ## field ## ]]` markers. It needs no special LM features — no JSON mode, no function calling, no native structured outputs. That breadth is why it’s the default. It also includes a safety net: when its regex parser fails, it falls back to JSONAdapter automatically (toggle with `use_json_adapter_fallback=False`).

### 3. Every adapter follows a fixed lifecycle

preprocess → format → LM call → postprocess → parse. You can debug any adapter by walking these five steps. Preprocess adapts the signature for native LM features (function calling, reasoning). Format builds the messages. The LM call returns raw outputs. Postprocess pulls out tool calls and native-typed responses (Reasoning, Citations). Parse runs `parse_value` on each declared output field.

### 4. Type coercion is centralized

`dspy/adapters/utils.py::parse_value` is the single function every adapter delegates to. Adapters differ in how they *find* a field’s value in the LM’s output (regex marker, JSON key, XML tag), but once they have the raw string they all call the same `parse_value(value, annotation)`. That keeps coercion rules consistent across adapter swaps and gives you one place to look when a typed field misbehaves.

### 5. Custom types own their own serialization

`Image`, `Audio`, `Code`, `Reasoning`, `Tool`, `ToolCalls`, and any user subclass of `dspy.Type` plug in their own `format()` and optional `parse_lm_response()`. The adapter doesn’t special-case images or audio. When the field’s annotation is a `dspy.Type` subclass, the adapter calls `type.format()` to render it into the provider’s content-block format and `type.parse_lm_response()` to read it back. Adding a new modality means writing a new subclass — no adapter changes.

### 6. Native LM features are surfaced through types

Function calling, structured outputs, and reasoning ride on `adapt_to_native_lm_feature()` and `parse_lm_response()` hooks on the type. A `dspy.Reasoning` output field, for example, tells the adapter “set `reasoning_effort` in `lm_kwargs` and pull the reasoning from `response['reasoning_content']` instead of regex-parsing it.” Model-specific feature integration lives at the type layer, where it’s reusable across adapters.

### 7. ChatAdapter falls back to JSONAdapter on parse error

Toggleable; on by default. The first time the LM produces malformed `[[ ## ## ]]` output, ChatAdapter catches the parse error and re-runs the request through JSONAdapter. This is more lenient than the right behavior for new code — you’d prefer to see the error — but it’s the default because it makes ChatAdapter usable against a wider range of models out of the box. Set `use_json_adapter_fallback=False` in tests.

### 8. JSONAdapter prefers structured output mode

It checks `lm.supported_params` and falls through tiers: OpenAI-style `response_format: json_schema` first, then `json_object` mode, then plain text JSON. The first tier is the most reliable because the model is constrained at decoding time, not only instructed.

### 9. TwoStepAdapter splits generation from extraction

For reasoning models that produce free-form text but format unreliably. Stage one: the main LM gets the task as plain prose, no field markers, and produces whatever shape it wants. Stage two: a smaller extractor LM with ChatAdapter reads the output and pulls out the declared fields. Useful for o1, o3-mini, and similar models where formatting reliability is the bottleneck.

### 10. Field marker format is hard-coded

`[[ ## name ## ]]` is the pattern, chosen for low collision and clean regex. The brackets-plus-hashes shape is unlikely to appear in real text or code, and the symmetry makes the parser simple. There’s no config knob to change it. JSONAdapter and XMLAdapter use their own formats; if you want a different chat-style format, subclass `Adapter`.

### 11. Finetuning data export is per-adapter

`format_finetune_data` is implemented on ChatAdapter (OpenAI message format); JSONAdapter raises `NotImplementedError`; TwoStepAdapter doesn’t support it either. If you’re using `BootstrapFinetune`, stay on ChatAdapter or implement `format_finetune_data` on the adapter of your choice.

## API walkthrough

Grouped by what you’re trying to do.

### The adapters

**`dspy.ChatAdapter(callbacks=None, use_native_function_calling=False, native_response_types=None, use_json_adapter_fallback=True)`**  
The default. Builds a chat-style prompt with field markers, parses the response with a regex over the same markers, and (by default) falls back to JSONAdapter if the regex misses. Set `use_json_adapter_fallback=False` when you want a hard error in tests.

**`dspy.JSONAdapter(callbacks=None, use_native_function_calling=True)`**  
Outputs structured JSON. Internally extends ChatAdapter — formatting is similar, but the output instruction asks for JSON and parsing uses `json_repair`. The constructor’s `use_native_function_calling=True` default flips when tool calling is wired in.

**`dspy.XMLAdapter(callbacks=None)`**  
`<field_name>value</field_name>` tags. The parser is a regex (`r"<(\w+)>(.*?)</\1>"` with `DOTALL`); it’s robust to whitespace but doesn’t tolerate nested tags of the same name.

**`dspy.TwoStepAdapter(extraction_model: BaseLM, **kwargs)`**  
Two LM calls per inference. Use it when the main LM is a reasoning model that’s bad at formatting — the extractor is usually a cheap general-purpose LM with ChatAdapter. Doesn’t support finetuning yet.

**`dspy.BAMLAdapter`**  
JSON-backed, but renders the output schema in BAML-style commented Pydantic form. Worth trying when JSONAdapter’s raw JSON schema is too verbose for complex nested types.

**`dspy.Adapter`**  
The base class. Subclass it when you want a new prompt shape — implement `format()`, `parse()`, and (optionally) `format_finetune_data()`.

### Adapter lifecycle

You’ll only override these methods when writing a custom adapter, but reading them helps when debugging a malformed prompt or a parse failure.

**`Adapter.__call__(lm, lm_kwargs, signature, demos, inputs)` / `Adapter.acall(...)`**  
Public entry. The flow inside is preprocess → format → LM call → postprocess. `lm_kwargs` (temperature, max_tokens, response_format, etc.) is where adapters reach during preprocess to request structured output or function calling.

**`Adapter.format(signature, demos, inputs)` → `list[dict]`**  
Turns the signature, demos, and inputs into chat messages. The pieces it composes from:

- `format_system_message(signature)` — the system message: field descriptions + format template + instructions.
- `format_field_description(signature)` — the per-field list with types and constraints.
- `format_field_structure(signature)` — the explanation of the marker format.
- `format_task_description(signature)` — `signature.instructions`.
- `format_demos(signature, demos)` — each demo becomes a user/assistant pair.
- `format_user_message_content(signature, inputs)` — the current call’s inputs.
- `format_assistant_message_content(signature, outputs)` — used inside demos.
- `format_conversation_history(signature, history)` — when a field has type `dspy.History`, expand it into turn messages instead of stuffing it into one field’s value.

**`Adapter.parse(signature, completion)` → `dict`**  
Extracts typed field values from the LM’s response. ChatAdapter regex-matches the marker pattern, splits by field, and delegates each value to `parse_value`. JSONAdapter parses the JSON object and pulls values by key. XMLAdapter walks tags.

**`Adapter.format_finetune_data(signature, demos, inputs, outputs)`**  
Serializes a demo into the LM provider’s finetune format. ChatAdapter writes OpenAI message format. Other adapters raise `NotImplementedError`.

### Type coercion

The single function every adapter calls when turning an LM-produced string into a typed value.

**`parse_value(value, annotation)` in `dspy/adapters/utils.py`**  
Strategy: if the annotation is `str`, pass through. If it’s an `Enum` or `Literal`, match against allowed values. Otherwise: try `json_repair.loads`, fall back to `ast.literal_eval`, fall back to the raw string; then validate with `TypeAdapter(annotation)`. If validation fails and the annotation is a `dspy.Type` subclass, retry with the raw value so the type’s own parser can take a shot.

Other helpers in the same file you’ll see in tracebacks:

- `format_field_value(field_info, value, assume_text=True)` — the inverse of parse: serializes a typed value for the prompt.
- `serialize_for_json(value)` — Pydantic-aware JSON serialization, used by JSONAdapter.
- `translate_field_type(field_info)` — generates the constraint string the prompt shows (“greater than: 0”).
- `get_field_description_string(fields)` — the field-list rendering.
- `find_enum_member(enum_cls, raw)` — resolves an enum by name or value.

### Custom type wrappers

Types adapters know how to render and parse beyond Python’s standard ones. Each implements `format()`; some implement `parse_lm_response()` and `adapt_to_native_lm_feature()` for native LM hooks.

**`dspy.adapters.types.Type`**  
The base class. Subclass it (it’s a `pydantic.BaseModel`) and implement `format()` to plug in a new type. Adapters wrap the output of `format()` with `<<CUSTOM-TYPE-START-IDENTIFIER>>...<<END-IDENTIFIER>>` so multi-modal content can be inserted into a single message stream and later split out.

**`dspy.Image(url, download=False, verify=True)`**  
URL, local path, bytes, or PIL image. `format()` returns the provider’s image content block (`{"type": "image_url", "image_url": {"url": ...}}`). `download=True` fetches the image and base64-encodes it, useful when the LM provider can’t reach the URL.

**`dspy.Audio(data, audio_format)`**  
Base64 audio data plus format string (`"wav"`, `"mp3"`). Renders as the provider’s audio content block.

**`dspy.File(file_data=None, file_id=None, filename=None)`**  
Either a data URI or a file ID (some providers preupload files and reference them by ID).

**`dspy.Code(code, language="python")`**  
Code with a class-level `language` parameter. `dspy.Code["java"]` produces a Code subclass typed for Java. `format()` returns the raw string — no wrapper, no fencing.

**`dspy.History(messages)`**  
Conversation turns. When the adapter sees this type on a field, it expands the messages into real user/assistant messages instead of stuffing them into one field’s value. Use this when you want the LM to see prior turns as messages.

**`dspy.Reasoning(content)`**  
String-like wrapper. When the LM supports a reasoning mode (o1, o3-mini, GPT-5 thinking variants), `Reasoning.adapt_to_native_lm_feature()` sets `reasoning_effort` in `lm_kwargs` and removes the field from the signature; `parse_lm_response()` pulls the reasoning out of the native response field. When the LM doesn’t support native reasoning, it falls back to a regular text field and behaves like the `reasoning` field `ChainOfThought` adds.

**`dspy.Tool(func, name=None, desc=None, args=None, arg_types=None, arg_desc=None)`**  
Wraps a Python callable. Auto-introspects the function signature if you don’t pass `args`/`arg_types`/`arg_desc`. Used by `ReAct` and anywhere modules accept tools. The full tool story lives in the Tools / ReAct / MCP DD page.

**`dspy.adapters.types.tool.ToolCalls.from_dict_list(...)`**  
The list of tool calls the LM produced, parsed from native function-calling responses.

**`dspy.adapters.types.Citations`**  
Declared as a default native response type. When the provider returns citations natively (e.g., Anthropic), adapters extract them through the type’s `parse_lm_response`.

### Configuring which adapter to use

- **`dspy.configure(adapter=dspy.JSONAdapter())`** — process-wide default.
- **`with dspy.context(adapter=dspy.XMLAdapter()): ...`** — scoped override.
- **No automatic LM-based selection.** ChatAdapter is the default and stays the default until you set otherwise. Some teleprompts (e.g., `BootstrapFinetune`) accept an `adapter` dict keyed by LM, so different LMs in a finetuning loop can use different adapters.

## Cross-links

- [Signatures in depth](signatures-in-depth.md) — what the adapter consumes.
- [Settings and context()](settings-and-context.md) — how `configure` and `context` propagate the adapter choice.
- Tools, ReAct, and MCP DD page — `Tool` and `ToolCalls` are adapter-formatted but module-driven.
