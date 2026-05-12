"""Tests for `to_tool_call`, the single boundary function that normalizes
every LiteLLM tool-call wire shape into a canonical `ToolCall`.

After this normalization, no downstream code should need to handle multiple
wire shapes. These tests pin down exactly which inputs the boundary accepts
and what it produces.
"""

import pytest

from dspy.adapters.types.tool import ToolCalls, to_tool_call

# ---------- Shape 1: OpenAI Chat Completions ----------

def test_chat_completions_dict_shape():
    item = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q":"hello"}'},
    }
    tc = to_tool_call(item)
    assert isinstance(tc, ToolCalls.ToolCall)
    assert tc.name == "search"
    assert tc.args == {"q": "hello"}
    assert tc.id == "call_abc"


def test_chat_completions_pydantic_shape():
    """Real LiteLLM `ChatCompletionMessageToolCall` is a pydantic object;
    we go through model_dump first."""
    class Fn:
        name = "search"
        arguments = '{"q":"x"}'

    class CCMToolCall:
        id = "call_123"
        type = "function"
        function = Fn()

        def model_dump(self):
            return {
                "id": self.id,
                "type": self.type,
                "function": {"name": self.function.name, "arguments": self.function.arguments},
            }

    tc = to_tool_call(CCMToolCall())
    assert tc.name == "search"
    assert tc.args == {"q": "x"}
    assert tc.id == "call_123"


def test_chat_completions_arguments_as_dict():
    """Some providers (and our own round-trips) put `arguments` as a dict."""
    item = {"type": "function", "function": {"name": "lookup", "arguments": {"k": "v"}}}
    tc = to_tool_call(item)
    assert tc.args == {"k": "v"}


def test_chat_completions_empty_arguments_string():
    """`arguments=""` should normalize to `{}`, not crash."""
    item = {"type": "function", "function": {"name": "ping", "arguments": ""}}
    assert to_tool_call(item).args == {}


# ---------- Shape 2: OpenAI Responses API ----------

def test_responses_api_dict_shape():
    item = {
        "type": "function_call",
        "name": "search",
        "arguments": '{"q":"y"}',
        "call_id": "call_xyz",
    }
    tc = to_tool_call(item)
    assert tc.name == "search"
    assert tc.args == {"q": "y"}
    assert tc.id == "call_xyz"


def test_responses_api_pydantic_shape():
    class FunctionCallItem:
        type = "function_call"
        name = "search"
        arguments = '{"q":"z"}'
        call_id = "call_99"

        def model_dump(self):
            return {
                "type": self.type,
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

    tc = to_tool_call(FunctionCallItem())
    assert tc.name == "search"
    assert tc.args == {"q": "z"}
    assert tc.id == "call_99"


# ---------- Shape 3: MockValSer / SchemaSerializer fallback ----------

def test_mockvalser_fallback_chat_completions_shape():
    """Cached LiteLLM pydantic whose model_dump raises TypeError — must fall
    back to attribute access via the `function` attribute.

    See https://github.com/pydantic/pydantic/issues/7713
    and https://github.com/BerriAI/litellm/issues/9345
    """
    class Fn:
        name = "search"
        arguments = '{"q":"cached"}'

    class CachedToolCall:
        id = "call_cached"
        type = "function"
        function = Fn()

        def model_dump(self):
            raise TypeError("'MockValSer' object cannot be converted to 'SchemaSerializer'")

    tc = to_tool_call(CachedToolCall())
    assert tc.name == "search"
    assert tc.args == {"q": "cached"}
    assert tc.id == "call_cached"


def test_mockvalser_fallback_no_recoverable_attrs_raises():
    """If model_dump fails AND there's no `function` or `name` attribute,
    re-raise the original MockValSer TypeError instead of returning garbage."""
    class Unsalvageable:
        def model_dump(self):
            raise TypeError("'MockValSer' object cannot be converted to 'SchemaSerializer'")

    with pytest.raises(TypeError, match="MockValSer"):
        to_tool_call(Unsalvageable())


# ---------- Error reporting ----------

def test_unknown_dict_shape_raises_with_payload():
    with pytest.raises(ValueError, match="Unknown tool-call shape"):
        to_tool_call({"unrelated": "data"})


def test_non_dict_non_pydantic_raises_with_type_info():
    with pytest.raises(TypeError, match="Cannot normalize tool call from int"):
        to_tool_call(42)


# ---------- ToolCall.id field ----------

def test_toolcall_id_field_optional():
    """ToolCall.id should default to None when not provided."""
    tc = ToolCalls.ToolCall(name="f", args={})
    assert tc.id is None


def test_toolcall_id_round_trips():
    tc = ToolCalls.ToolCall(name="f", args={"a": 1}, id="call_zzz")
    assert tc.id == "call_zzz"
