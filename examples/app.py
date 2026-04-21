"""Streamlit demo: DSPy with Apple on-device language models.

Usage:
    streamlit run examples/apple_lm_streamlit_app.py

Requirements (one or both):
    pip install apple-fm-sdk   # AppleFoundationLM — macOS 26+ with Apple Intelligence
    pip install mlx-lm         # AppleLocalLM      — Apple Silicon Mac
"""

from __future__ import annotations

import ast
import io
import logging
import operator as op
import sys
import traceback
from typing import Literal

import streamlit as st

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

import dspy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSPy · Apple LM Demo",
    layout="wide",
)


# ── Safe arithmetic evaluator for ReAct demo ─────────────────────────────────
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. The expression argument must be a plain math string
    such as '15 * 240 / 100' or '(3 + 4) ** 2'. Do NOT pass a schema or description — pass the
    actual expression you want computed. Returns the numeric result as a string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def wikipedia_summary(topic: str) -> str:
    """Return a short description of a well-known topic. The topic argument must be a plain text
    string such as 'Python' or 'Apple Silicon' — not a schema or JSON object. Only use this tool
    for factual lookups, not arithmetic."""
    stubs: dict[str, str] = {
        "dspy": "DSPy is a framework for algorithmically optimising LM prompts and weights.",
        "apple silicon": "Apple Silicon is Apple's line of ARM-based SoC processors (M1, M2, M3, M4).",
        "mlx": "MLX is Apple's array framework for machine learning on Apple Silicon.",
        "python": "Python is a high-level, dynamically typed, general-purpose programming language.",
        "llama": "LLaMA is a family of open-weight large language models released by Meta AI.",
    }
    key = topic.strip().lower()
    for k, v in stubs.items():
        if k in key:
            return v
    return f"No stub found for '{topic}'. (This is a demo — not real Wikipedia.)"


# ── Structured output schema ──────────────────────────────────────────────────
class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: int = Field(ge=1, le=10, description="Confidence from 1 (low) to 10 (high)")
    key_phrases: list[str] = Field(description="Up to 3 key phrases that drove the sentiment")


class SentimentSignature(dspy.Signature):
    """Analyse the sentiment of the provided text and return a structured result."""

    text: str = dspy.InputField()
    result: SentimentResult = dspy.OutputField()


# ── DSPy history helper ───────────────────────────────────────────────────────
def _show_history() -> None:
    with st.expander("DSPy history (last call)", expanded=False):
        try:
            buf = io.StringIO()
            _prev, sys.stdout = sys.stdout, buf
            dspy.inspect_history(n=1)
            sys.stdout = _prev
            output = buf.getvalue().strip()
            st.code(output or "(no history yet)", language="text")
        except Exception as exc:
            st.text(f"History unavailable: {exc}")


# ── Sidebar: LM configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.title("Model Config")

    lm_type = st.selectbox(
        "Language model",
        ["AppleFoundationLM", "AppleLocalLM"],
        help=(
            "**AppleFoundationLM** — Apple Intelligence system model. "
            "Requires macOS 26+ and `pip install apple-fm-sdk`.\n\n"
            "**AppleLocalLM** — Any mlx-lm model from HuggingFace. "
            "Requires Apple Silicon and `pip install mlx-lm`."
        ),
    )

    if lm_type == "AppleLocalLM":
        model_name = st.text_input(
            "Model (HuggingFace repo ID or local path)",
            value="mlx-community/Llama-3.2-3B-Instruct-4bit",
            help="Browse pre-quantized models at huggingface.co/mlx-community",
        )
        bits = st.selectbox(
            "Quantization hint",
            [None, 4, 8],
            format_func=lambda x: "auto-detect" if x is None else f"{x}-bit",
            help="Informational only — does not trigger automatic quantization.",
        )
    else:
        model_name = "apple/on-device"
        bits = None

    st.divider()

    default_temp = 0.0 if lm_type == "AppleLocalLM" else 0.7
    temperature = st.slider("Temperature", 0.0, 1.5, value=default_temp, step=0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, step=64)
    cache = st.checkbox("Enable DSPy cache", value=True)

    st.divider()

    if st.button("Initialize LM", type="primary", use_container_width=True):
        st.session_state.lm_loading = True
        st.session_state.pop("lm_init_error", None)
        with st.spinner("Loading…"):
            try:
                if lm_type == "AppleFoundationLM":
                    lm = dspy.AppleFoundationLM(
                        temperature=temperature if temperature > 0 else None,
                        max_tokens=max_tokens,
                        cache=cache,
                    )
                else:
                    lm = dspy.AppleLocalLM(
                        model=model_name,
                        bits=bits,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        cache=cache,
                    )
                st.session_state.lm = lm
                st.session_state.lm_label = (
                    "AppleFoundationLM"
                    if lm_type == "AppleFoundationLM"
                    else f"AppleLocalLM · {model_name}"
                )
                st.success("Ready")
            except ImportError as exc:
                pkg = "apple-fm-sdk" if lm_type == "AppleFoundationLM" else "mlx-lm"
                msg = f"Missing package — run: `pip install {pkg}`\n\n{exc}"
                logger.error(msg)
                st.session_state.lm_init_error = msg
                st.error(msg)
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                st.session_state.lm_init_error = tb
                st.error(tb)
            finally:
                st.session_state.lm_loading = False

    if st.session_state.get("lm_init_error"):
        st.error(st.session_state.lm_init_error)

    if "lm_label" in st.session_state:
        st.info(f"Active: {st.session_state.lm_label}")


# ── Guard ─────────────────────────────────────────────────────────────────────
st.title("DSPy · Apple LM Demo")

if "lm" not in st.session_state:
    if not st.session_state.get("lm_loading"):
        st.info(
            "Select a model and click **Initialize LM** in the sidebar to get started.\n\n"
            "| Backend | Requirement |\n"
            "|---|---|\n"
            "| **AppleFoundationLM** | macOS 26+ · Apple Intelligence · `pip install apple-fm-sdk` |\n"
            "| **AppleLocalLM** | Apple Silicon · `pip install mlx-lm` |"
        )
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_cot, tab_react, tab_struct, tab_custom = st.tabs([
    "Predict",
    "Chain of Thought",
    "ReAct",
    "Structured Output",
    "Custom Signature",
])


# ── Predict ───────────────────────────────────────────────────────────────────
with tab_predict:
    st.header("dspy.Predict")
    st.caption(
        "`Predict` is the simplest DSPy module. "
        "It sends a single forward pass through the LM using your signature."
    )
    with st.form("form_predict"):
        question = st.text_area(
            "Question",
            "What are the main benefits of running AI on-device?",
            height=80,
        )
        submitted = st.form_submit_button("Run Predict", type="primary")

    if submitted:
        with st.spinner("Thinking…"):
            try:
                with dspy.context(lm=st.session_state.lm):
                    result = dspy.Predict("question -> answer")(question=question)
                st.subheader("Answer")
                st.write(result.answer)
                _show_history()
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                st.error(tb)


# ── Chain of Thought ──────────────────────────────────────────────────────────
with tab_cot:
    st.header("dspy.ChainOfThought")
    st.caption(
        "`ChainOfThought` prepends a `reasoning` field to the signature, "
        "letting the model work through a problem step-by-step before producing an answer."
    )
    with st.form("form_cot"):
        question_cot = st.text_area(
            "Question",
            "A bat and a ball together cost $1.10. "
            "The bat costs $1.00 more than the ball. How much does the ball cost?",
            height=100,
        )
        submitted_cot = st.form_submit_button("Run Chain of Thought", type="primary")

    if submitted_cot:
        with st.spinner("Reasoning…"):
            try:
                with dspy.context(lm=st.session_state.lm):
                    result = dspy.ChainOfThought("question -> answer")(question=question_cot)
                col_r, col_a = st.columns([3, 2])
                with col_r:
                    st.subheader("Reasoning")
                    st.info(result.reasoning)
                with col_a:
                    st.subheader("Answer")
                    st.success(result.answer)
                _show_history()
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                st.error(tb)


# ── ReAct ─────────────────────────────────────────────────────────────────────
with tab_react:
    st.header("dspy.ReAct")
    st.caption(
        "`ReAct` (Reasoning + Acting) lets the model call tools iteratively. "
        "Available tools: **calculator** (safe arithmetic), **wikipedia_summary** (stub lookup)."
    )
    with st.expander("Tool definitions", expanded=False):
        st.code(
            "def calculator(expression: str) -> str:\n"
            "    \"\"\"Evaluate a safe arithmetic expression (e.g. '15 * 240 / 100').\"\"\"\n"
            "    ...\n\n"
            "def wikipedia_summary(topic: str) -> str:\n"
            "    \"\"\"Return a short description of a well-known topic.\"\"\"\n"
            "    ...",
            language="python",
        )
    with st.form("form_react"):
        question_react = st.text_area(
            "Question",
            "What is 15% of 240? Use the calculator tool to compute it.",
            height=80,
            help="Tip: keep questions focused on arithmetic (calculator) or simple factual topics "
                 "(wikipedia_summary). Small on-device models may struggle with open-ended queries.",
        )
        max_iters = st.slider("Max iterations", 1, 10, 5)
        submitted_react = st.form_submit_button("Run ReAct", type="primary")

    if submitted_react:
        with st.spinner("Acting…"):
            try:
                with dspy.context(lm=st.session_state.lm):
                    react = dspy.ReAct(
                        "question -> answer",
                        tools=[calculator, wikipedia_summary],
                        max_iters=max_iters,
                    )
                    result = react(question=question_react)
                st.subheader("Answer")
                st.write(result.answer)
                _show_history()
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                st.error(tb)


# ── Structured Output ─────────────────────────────────────────────────────────
with tab_struct:
    st.header("dspy.Predict · Structured Output")
    st.caption(
        "Uses a typed `dspy.Signature` with a Pydantic output field. "
        "DSPy asks the model to produce a `result` key containing the `SentimentResult` schema. "
        "AppleFoundationLM additionally applies native constrained decoding."
    )
    st.info(
        "**Schema: `SentimentResult`**\n"
        "- `sentiment`: `positive` | `negative` | `neutral`\n"
        "- `confidence`: integer 1–10\n"
        "- `key_phrases`: list of strings"
    )
    with st.form("form_struct"):
        text_input = st.text_area(
            "Text to analyse",
            "Apple Inc. reported record quarterly revenue of $124.3 billion on Tuesday, "
            "driven by strong iPhone 17 sales and Services growth. CEO Tim Cook called it "
            "'the best quarter in Apple's history.'",
            height=120,
        )
        submitted_struct = st.form_submit_button("Run Structured Predict", type="primary")

    if submitted_struct:
        with st.spinner("Extracting…"):
            try:
                with dspy.context(lm=st.session_state.lm):
                    extractor = dspy.Predict(SentimentSignature)
                    prediction = extractor(text=text_input)
                raw = prediction.result

                # Normalise — backend may return a model instance, dict, or JSON string
                data: SentimentResult | None = None
                parse_error: str | None = None
                if isinstance(raw, SentimentResult):
                    data = raw
                elif isinstance(raw, dict):
                    try:
                        data = SentimentResult(**raw)
                    except Exception as exc:
                        parse_error = str(exc)
                else:
                    import re
                    # Try to extract a JSON object from the raw string (model may wrap it in prose)
                    json_match = re.search(r"\{.*\}", str(raw), re.DOTALL)
                    if json_match:
                        try:
                            data = SentimentResult.model_validate_json(json_match.group())
                        except Exception as exc:
                            parse_error = str(exc)
                    else:
                        parse_error = f"No JSON object found in model output."

                if data is not None:
                    emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Sentiment", f"{emoji.get(data.sentiment, '')} {data.sentiment}")
                        st.metric("Confidence", f"{data.confidence} / 10")
                    with col2:
                        st.subheader("Key Phrases")
                        for phrase in data.key_phrases:
                            st.write(f"• {phrase}")
                else:
                    st.warning(
                        f"Could not parse structured output ({parse_error}). "
                        "This model may not reliably follow JSON schemas — try AppleFoundationLM "
                        "for native constrained decoding."
                    )
                    st.subheader("Raw model output")
                    st.code(str(raw), language="text")
                _show_history()
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                st.error(tb)


# ── Custom Signature ──────────────────────────────────────────────────────────
with tab_custom:
    st.header("Custom Signature")
    st.caption(
        "Define any DSPy signature (`inputs -> outputs`), fill in the input values, "
        "and run with either `Predict` or `ChainOfThought`."
    )

    sig_str = st.text_input(
        "Signature",
        "article -> tickers, sentiment",
        help="Format: `input1, input2 -> output1, output2`",
    )

    # Parse signature
    try:
        lhs, rhs = sig_str.split("->")
        input_fields = [f.strip() for f in lhs.split(",") if f.strip()]
        output_fields = [f.strip() for f in rhs.split(",") if f.strip()]
        sig_valid = bool(input_fields and output_fields)
    except ValueError:
        input_fields, output_fields, sig_valid = [], [], False

    if not sig_valid:
        st.warning("Signature must be in the format `input1, input2 -> output1`.")
    else:
        st.write(
            f"**Inputs:** {', '.join(f'`{f}`' for f in input_fields)}"
            f"  →  **Outputs:** {', '.join(f'`{f}`' for f in output_fields)}"
        )

        with st.form("form_custom"):
            input_values: dict[str, str] = {}
            for field in input_fields:
                input_values[field] = st.text_area(f"`{field}`", key=f"ci_{field}", height=80)

            module_choice = st.radio(
                "Module", ["Predict", "ChainOfThought"], horizontal=True
            )
            submitted_custom = st.form_submit_button("Run", type="primary")

        if submitted_custom:
            with st.spinner("Running…"):
                try:
                    with dspy.context(lm=st.session_state.lm):
                        mod = (
                            dspy.ChainOfThought(sig_str)
                            if module_choice == "ChainOfThought"
                            else dspy.Predict(sig_str)
                        )
                        result = mod(**input_values)

                    if module_choice == "ChainOfThought" and hasattr(result, "reasoning"):
                        st.subheader("Reasoning")
                        st.info(result.reasoning)

                    st.subheader("Outputs")
                    for field in output_fields:
                        val = getattr(result, field, None)
                        if val is not None:
                            st.write(f"**{field}:** {val}")
                    _show_history()
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(tb)
                    st.error(tb)
