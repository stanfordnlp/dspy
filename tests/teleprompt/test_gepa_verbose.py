"""
tests/teleprompt/test_gepa_verbose.py

测试目标：验证 GEPA verbose=True 参数能正确向 stdout 输出：
  1. 反思的 Prompt（发给 reflection_lm 的完整 prompt）
  2. 反思的结果（reflection_lm 的原始返回）
  3. 优化后的新指令

测试策略：
  - 用 capsys 捕获 stdout（pytest 内置 fixture，无需联网）
  - 用 CapturingLM 作为 reflection_lm，固定返回内容，验证输出内容可预期
  - 验证 verbose=False（默认）时不打印任何内容（不影响原有行为）

运行方式：
  pytest tests/teleprompt/test_gepa_verbose.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import dspy
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM

# ─────────────────────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────────────────────

MOCK_NEW_INSTRUCTION = "This is the improved instruction returned by reflection LM."


class CapturingLM:
    """
    伪造的 reflection_lm：
    - 返回固定的 MOCK_NEW_INSTRUCTION（wrapped 在 ``` 代码块里，符合解析规范）
    - 同时记录收到的 prompt 用于额外断言
    """

    def __init__(self):
        self.received_prompts: list[str] = []

    def __call__(self, prompt, **kwargs):
        self.received_prompts.append(str(prompt))
        return [f"```\n{MOCK_NEW_INSTRUCTION}\n```"]


class SimpleQAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predict("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


def always_wrong_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """永远返回 0 分，确保 GEPA 触发反思流程"""
    return dspy.Prediction(score=0, feedback="Wrong. Please improve the instruction.")


def _run_propose(verbose: bool, capsys):
    """
    直接调用 DspyAdapter.propose_new_texts，
    返回 (capturing_lm, results, stdout_text)
    """
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    student = SimpleQAModule()
    capturing_lm = CapturingLM()

    adapter = DspyAdapter(
        student_module=student,
        metric_fn=always_wrong_metric,
        feedback_map={},
        reflection_lm=capturing_lm,
        verbose=verbose,
    )

    candidate = {"predictor": "Answer the question."}
    reflective_dataset = {
        "predictor": [
            {
                "Inputs": {"question": "What is 2+2?"},
                "Generated Outputs": {"answer": "5"},
                "Feedback": "Wrong. The correct answer is 4.",
            }
        ]
    }

    with dspy.context(lm=DummyLM([{"answer": "dummy"}])):
        results = adapter.propose_new_texts(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=["predictor"],
        )

    captured = capsys.readouterr()
    return capturing_lm, results, captured.out


# ─────────────────────────────────────────────────────────────────────────────
# 第 1 组：verbose=False（默认）时没有任何 stdout 输出
# ─────────────────────────────────────────────────────────────────────────────


class TestVerboseFalse:
    """verbose=False 时不打印任何内容，保持原有行为"""

    def test_no_stdout_when_verbose_false(self, capsys):
        """verbose=False 时 stdout 为空"""
        _, _, stdout = _run_propose(verbose=False, capsys=capsys)
        assert stdout == "", f"verbose=False 时不应有任何 stdout 输出，但实际输出：\n{stdout}"

    def test_result_still_correct_when_verbose_false(self, capsys):
        """verbose=False 时结果仍然正确返回"""
        _, results, _ = _run_propose(verbose=False, capsys=capsys)
        assert "predictor" in results
        assert results["predictor"] == MOCK_NEW_INSTRUCTION

    def test_reflection_counter_increments_when_verbose_false(self, capsys):
        """verbose=False 时计数器仍正常递增"""
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        student = SimpleQAModule()
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=always_wrong_metric,
            feedback_map={},
            reflection_lm=CapturingLM(),
            verbose=False,
        )
        assert adapter._reflection_call_count == 0

        with dspy.context(lm=DummyLM([{"answer": "dummy"}])):
            adapter.propose_new_texts(
                candidate={"predictor": "instruction"},
                reflective_dataset={
                    "predictor": [
                        {
                            "Inputs": {"question": "q"},
                            "Generated Outputs": {"answer": "a"},
                            "Feedback": "feedback",
                        }
                    ]
                },
                components_to_update=["predictor"],
            )

        assert adapter._reflection_call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 第 2 组：verbose=True 时 stdout 包含反思 Prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestVerboseTruePromptOutput:
    """验证 verbose=True 时，反思 Prompt 被正确打印"""

    def test_prompt_header_printed(self, capsys):
        """stdout 包含反思 Prompt 的标题行"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "发给 reflection_lm 的 Prompt" in stdout, f"stdout 中未找到 Prompt 标题。\nstdout:\n{stdout}"

    def test_prompt_separator_printed(self, capsys):
        """stdout 包含用于分隔的 === 分隔线"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "=" * 70 in stdout, f"stdout 中未找到 === 分隔线。\nstdout:\n{stdout}"

    def test_prompt_contains_current_instruction(self, capsys):
        """stdout 中的 Prompt 包含当前 instruction 原文"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "Answer the question." in stdout, f"stdout 中未找到当前 instruction。\nstdout:\n{stdout}"

    def test_prompt_contains_feedback(self, capsys):
        """stdout 中的 Prompt 包含 feedback 内容"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "Wrong. The correct answer is 4." in stdout, f"stdout 中未找到 feedback 内容。\nstdout:\n{stdout}"

    def test_prompt_contains_reflection_call_count(self, capsys):
        """stdout 中包含反思次数计数（第 1 次）"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "第 1 次反思" in stdout, f"stdout 中未找到反思次数计数。\nstdout:\n{stdout}"


# ─────────────────────────────────────────────────────────────────────────────
# 第 3 组：verbose=True 时 stdout 包含反思结果（reflection_lm 的返回）
# ─────────────────────────────────────────────────────────────────────────────


class TestVerboseTrueResultOutput:
    """验证 verbose=True 时，reflection_lm 的返回被正确打印"""

    def test_result_header_printed(self, capsys):
        """stdout 包含反思结果的标题行"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "reflection_lm 原始返回" in stdout, f"stdout 中未找到结果标题。\nstdout:\n{stdout}"

    def test_result_separator_printed(self, capsys):
        """stdout 包含 --- 分隔线"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "─" * 70 in stdout, f"stdout 中未找到 --- 分隔线。\nstdout:\n{stdout}"

    def test_raw_lm_output_in_stdout(self, capsys):
        """reflection_lm 返回的原始内容（含 ``` 代码块）出现在 stdout 中"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert MOCK_NEW_INSTRUCTION in stdout, (
            f"stdout 中未找到 reflection_lm 返回的原始内容。\n期望包含：{MOCK_NEW_INSTRUCTION}\n实际 stdout：\n{stdout}"
        )

    def test_new_instruction_header_printed(self, capsys):
        """stdout 包含优化后新指令的标题行"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "优化后新指令" in stdout, f"stdout 中未找到优化后新指令标题。\nstdout:\n{stdout}"

    def test_new_instruction_content_in_stdout(self, capsys):
        """解析后的新指令内容出现在 stdout 中"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert MOCK_NEW_INSTRUCTION in stdout, (
            f"stdout 中未找到新指令内容。\n期望包含：{MOCK_NEW_INSTRUCTION}\n实际 stdout：\n{stdout}"
        )

    def test_component_name_in_stdout(self, capsys):
        """stdout 中包含组件名称（predictor）"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        assert "predictor" in stdout, f"stdout 中未找到组件名称。\nstdout:\n{stdout}"


# ─────────────────────────────────────────────────────────────────────────────
# 第 4 组：输出顺序验证
# ─────────────────────────────────────────────────────────────────────────────


class TestVerboseOutputOrder:
    """验证输出顺序正确：先 Prompt，后结果，最后新指令"""

    def test_prompt_before_result(self, capsys):
        """'发给 reflection_lm 的 Prompt' 出现在 'reflection_lm 原始返回' 之前"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        pos_prompt = stdout.find("发给 reflection_lm 的 Prompt")
        pos_result = stdout.find("reflection_lm 原始返回")
        assert pos_prompt != -1 and pos_result != -1
        assert pos_prompt < pos_result, "Prompt 标题应该出现在结果标题之前"

    def test_result_before_new_instruction(self, capsys):
        """'reflection_lm 原始返回' 出现在 '优化后新指令' 之前"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        pos_result = stdout.find("reflection_lm 原始返回")
        pos_new = stdout.find("优化后新指令")
        assert pos_result != -1 and pos_new != -1
        assert pos_result < pos_new, "结果标题应该出现在新指令标题之前"

    def test_full_output_order(self, capsys):
        """完整顺序：Prompt 标题 → Prompt 内容 → 结果标题 → 新指令标题"""
        _, _, stdout = _run_propose(verbose=True, capsys=capsys)
        markers = [
            "发给 reflection_lm 的 Prompt",
            "Answer the question.",  # Prompt 内容
            "reflection_lm 原始返回",
            MOCK_NEW_INSTRUCTION,  # 结果内容
            "优化后新指令",
        ]
        positions = []
        for m in markers:
            pos = stdout.find(m)
            assert pos != -1, f"stdout 中未找到：{m}"
            positions.append(pos)

        assert positions == sorted(positions), f"输出顺序不符合预期。各标记位置：{list(zip(markers, positions))}"


# ─────────────────────────────────────────────────────────────────────────────
# 第 5 组：多次反思时计数器递增
# ─────────────────────────────────────────────────────────────────────────────


class TestVerboseMultipleReflections:
    """验证多次调用时，反思计数器正确递增，每次都有对应输出"""

    def test_counter_increments_across_calls(self, capsys):
        """两次调用 propose_new_texts，计数器从 1 递增到 2"""
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        student = SimpleQAModule()
        capturing_lm = CapturingLM()

        adapter = DspyAdapter(
            student_module=student,
            metric_fn=always_wrong_metric,
            feedback_map={},
            reflection_lm=capturing_lm,
            verbose=True,
        )

        candidate = {"predictor": "Initial instruction."}
        reflective_dataset = {
            "predictor": [
                {
                    "Inputs": {"question": "q"},
                    "Generated Outputs": {"answer": "wrong"},
                    "Feedback": "feedback text",
                }
            ]
        }

        with dspy.context(lm=DummyLM([{"answer": "dummy"}])):
            adapter.propose_new_texts(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=["predictor"],
            )
            adapter.propose_new_texts(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=["predictor"],
            )

        stdout = capsys.readouterr().out
        assert "第 1 次反思" in stdout, "未找到第 1 次反思"
        assert "第 2 次反思" in stdout, "未找到第 2 次反思"
        assert adapter._reflection_call_count == 2

    def test_both_prompts_appear_for_two_calls(self, capsys):
        """两次调用，两个 Prompt 都出现在 stdout 中"""
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        student = SimpleQAModule()

        adapter = DspyAdapter(
            student_module=student,
            metric_fn=always_wrong_metric,
            feedback_map={},
            reflection_lm=CapturingLM(),
            verbose=True,
        )

        candidate = {"predictor": "instruction v1"}
        reflective_dataset = {
            "predictor": [
                {
                    "Inputs": {"question": "q"},
                    "Generated Outputs": {"answer": "a"},
                    "Feedback": "feedback",
                }
            ]
        }

        with dspy.context(lm=DummyLM([{"answer": "dummy"}])):
            adapter.propose_new_texts(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=["predictor"],
            )
            adapter.propose_new_texts(
                candidate={"predictor": "instruction v2"},
                reflective_dataset=reflective_dataset,
                components_to_update=["predictor"],
            )

        stdout = capsys.readouterr().out
        # 两次 Prompt 都应出现
        assert stdout.count("发给 reflection_lm 的 Prompt") == 2
        assert stdout.count("reflection_lm 原始返回") == 2
        assert stdout.count("优化后新指令") == 2
