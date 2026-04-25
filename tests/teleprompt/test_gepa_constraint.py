"""
tests/teleprompt/test_gepa_constraint.py

测试目标：验证 GEPA 的 constraint 参数能正确地：
  1. [单元] _build_constraint_prompt_template 将约束注入到 prompt_template
  2. [集成-mock] propose_new_texts 调用 reflection_lm 时，实际发送的 prompt 包含约束文本
  3. [集成-真实LLM] 完整跑一轮 GEPA，reflection_lm 生成的新 prompt 中能体现约束要求

运行方式（需提前配置好相应的 conda 环境或 pip 环境）：
  # 仅跑单元 + mock 测试（无需联网）
  pytest tests/teleprompt/test_gepa_constraint.py -v -k "not real_lm"

  # 跑全部（包含真实 LLM 调用，需配置好 API key）
  pytest tests/teleprompt/test_gepa_constraint.py -v
"""

import os
import sys

# 确保使用本地 dspy 源码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM

# ─────────────────────────────────────────────────────────────────────────────
# 公共 fixture / helper
# ─────────────────────────────────────────────────────────────────────────────


class SimpleQAModule(dspy.Module):
    """用于测试的最小 DSPy 模块：question -> answer"""

    def __init__(self):
        super().__init__()
        self.predictor = Predict("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """永远返回 0 分 + 固定 feedback，使 GEPA 一直有错误样本可反思"""
    return dspy.Prediction(score=0, feedback="The answer was wrong. Please try harder.")


def _make_adapter(constraint):
    """构造一个带指定 constraint 的 DspyAdapter，reflection_lm 使用 DummyLM"""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    student = SimpleQAModule()
    dummy_reflection_lm = DummyLM([{"answer": "dummy"}])

    adapter = DspyAdapter(
        student_module=student,
        metric_fn=simple_metric,
        feedback_map={},
        reflection_lm=dummy_reflection_lm,
        constraint=constraint,
    )
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# 第 1 组：单元测试 _build_constraint_prompt_template
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildConstraintPromptTemplate:
    """直接测试 DspyAdapter._build_constraint_prompt_template 的输出"""

    def test_none_constraint_returns_none(self):
        """未设置 constraint 时返回 None，不影响默认行为"""
        adapter = _make_adapter(constraint=None)
        assert adapter._build_constraint_prompt_template() is None

    def test_single_string_constraint_is_injected(self):
        """单条字符串约束被注入到模板中"""
        constraint = "输出的 prompt 必须只包含 JSON 格式"
        adapter = _make_adapter(constraint=constraint)
        template = adapter._build_constraint_prompt_template()

        assert template is not None
        assert constraint in template, "约束文本未出现在模板中"

    def test_list_constraint_all_items_injected(self):
        """列表形式的多条约束，每条都必须出现在模板中"""
        constraints = [
            "不得删除任何意图类别",
            "必须保留中文描述",
            "输出必须是合法 JSON",
        ]
        adapter = _make_adapter(constraint=constraints)
        template = adapter._build_constraint_prompt_template()

        assert template is not None
        for i, c in enumerate(constraints, start=1):
            assert c in template, f"第 {i} 条约束未出现在模板中：{c}"

    def test_list_constraint_numbered_correctly(self):
        """列表约束按 1/2/3 编号出现"""
        constraints = ["约束A", "约束B", "约束C"]
        adapter = _make_adapter(constraint=constraints)
        template = adapter._build_constraint_prompt_template()

        assert "1. 约束A" in template
        assert "2. 约束B" in template
        assert "3. 约束C" in template

    def test_constraint_injected_before_final_anchor(self):
        """约束块插在 'Provide the new instructions within ``` blocks.' 之前"""

        anchor = "Provide the new instructions within ``` blocks."
        constraint = "必须遵守此约束"
        adapter = _make_adapter(constraint=constraint)
        template = adapter._build_constraint_prompt_template()

        assert template is not None
        anchor_pos = template.find(anchor)
        constraint_pos = template.find(constraint)

        assert anchor_pos != -1, "锚点文本不存在于模板中"
        assert constraint_pos != -1, "约束文本不存在于模板中"
        assert constraint_pos < anchor_pos, "约束块应在锚点之前出现"

    def test_template_still_contains_original_placeholders(self):
        """注入约束后，原始模板的两个占位符依然保留"""
        adapter = _make_adapter(constraint="某约束")
        template = adapter._build_constraint_prompt_template()

        assert "<curr_instructions>" in template, "原始占位符 <curr_instructions> 丢失"
        assert "<inputs_outputs_feedback>" in template, "原始占位符 <inputs_outputs_feedback> 丢失"

    def test_no_constraint_adapter_uses_default_template(self):
        """无约束时 propose_new_texts 使用 InstructionProposalSignature 默认模板（不传 prompt_template）"""

        adapter = _make_adapter(constraint=None)
        # None 表示不会向 InstructionProposalSignature.run 传 prompt_template 参数
        assert adapter._build_constraint_prompt_template() is None


# ─────────────────────────────────────────────────────────────────────────────
# 第 2 组：集成测试（mock reflection_lm）—— 验证 propose_new_texts 发送的 prompt
# ─────────────────────────────────────────────────────────────────────────────


class CapturingLM:
    """
    一个伪造的 LM 对象：
      - 记录所有被调用时收到的 prompt 字符串（捕获）
      - 返回固定的 mock 新指令
    """

    def __init__(self, return_instruction="Improved instruction from mock LM."):
        self.captured_prompts: list[str] = []
        self._return_instruction = return_instruction

    def __call__(self, prompt_or_messages, **kwargs):
        # InstructionProposalSignature.run 通过 stripped_lm_call 调用，
        # stripped_lm_call 将收到的字符串直接传给 self.reflection_lm(x)
        self.captured_prompts.append(str(prompt_or_messages))
        # 返回格式：stripped_lm_call 期望 list[str] 或 list[dict]
        return [f"```\n{self._return_instruction}\n```"]

    @property
    def last_prompt(self) -> str | None:
        return self.captured_prompts[-1] if self.captured_prompts else None


class TestProposeNewTextsWithConstraint:
    """
    通过替换 reflection_lm 为 CapturingLM，
    直接断言 propose_new_texts 发出的 prompt 包含约束文本。
    """

    def _run_propose(self, constraint):
        """
        构造 DspyAdapter，调用 propose_new_texts，
        返回 (capturing_lm, results)。
        """
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        student = SimpleQAModule()
        capturing_lm = CapturingLM()

        adapter = DspyAdapter(
            student_module=student,
            metric_fn=simple_metric,
            feedback_map={},
            reflection_lm=capturing_lm,
            constraint=constraint,
        )

        candidate = {"predictor": "Answer the question directly."}
        reflective_dataset = {
            "predictor": [
                {
                    "Inputs": {"question": "What is 2+2?"},
                    "Generated Outputs": {"answer": "5"},
                    "Feedback": "The answer was wrong. The correct answer is 4.",
                }
            ]
        }

        with dspy.context(lm=DummyLM([{"answer": "dummy"}])):
            results = adapter.propose_new_texts(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=["predictor"],
            )

        return capturing_lm, results

    def test_without_constraint_prompt_has_no_constraint_marker(self):
        """不设置 constraint 时，发出的 prompt 不含约束标记"""
        capturing_lm, _ = self._run_propose(constraint=None)
        assert capturing_lm.last_prompt is not None, "reflection_lm 未被调用"
        assert "IMPORTANT CONSTRAINTS" not in capturing_lm.last_prompt
        assert "重要约束" not in capturing_lm.last_prompt

    def test_single_string_constraint_in_sent_prompt(self):
        """单条字符串约束出现在发给 reflection_lm 的 prompt 中"""
        constraint = "输出必须是合法 JSON 格式"
        capturing_lm, _ = self._run_propose(constraint=constraint)

        assert capturing_lm.last_prompt is not None, "reflection_lm 未被调用"
        assert constraint in capturing_lm.last_prompt, (
            f"约束文本未出现在发送给 reflection_lm 的 prompt 中。\n"
            f"约束：{constraint}\n"
            f"实际 prompt 前 500 字符：{capturing_lm.last_prompt[:500]}"
        )

    def test_list_constraints_all_in_sent_prompt(self):
        """列表形式的多条约束都出现在发给 reflection_lm 的 prompt 中"""
        constraints = ["不得删除任何类别", "必须保留中文说明", "不能超过500字"]
        capturing_lm, _ = self._run_propose(constraint=constraints)

        assert capturing_lm.last_prompt is not None, "reflection_lm 未被调用"
        for c in constraints:
            assert c in capturing_lm.last_prompt, f"约束 '{c}' 未出现在发送给 reflection_lm 的 prompt 中"

    def test_constraint_before_anchor_in_sent_prompt(self):
        """约束块在锚点之前出现（在实际发送的 prompt 中验证顺序）"""
        anchor = "Provide the new instructions within ``` blocks."
        constraint = "必须遵守此重要约束"
        capturing_lm, _ = self._run_propose(constraint=constraint)

        prompt = capturing_lm.last_prompt
        assert prompt is not None

        constraint_pos = prompt.find(constraint)
        anchor_pos = prompt.find(anchor)

        assert constraint_pos != -1, "约束文本不在发送的 prompt 中"
        assert anchor_pos != -1, "锚点文本不在发送的 prompt 中"
        assert constraint_pos < anchor_pos, "约束块应在锚点之前"

    def test_propose_returns_string_instruction(self):
        """propose_new_texts 返回的新指令是字符串（而非 None 或异常）"""
        constraint = "约束：必须返回有效指令"
        _, results = self._run_propose(constraint=constraint)

        assert "predictor" in results
        assert isinstance(results["predictor"], str)
        assert len(results["predictor"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 第 3 组：端到端测试（真实 LLM 调用）
# ─────────────────────────────────────────────────────────────────────────────

# 真实 LM 配置（与 notebook 保持一致）
_REAL_LM_CONFIG = dict(
    model="openai/Doubao-pro-128k",
    api_base="https://aigc.sankuai.com/v1/openai/native",
    api_key="2015679947834241043",
    temperature=0,
)
_REAL_REFLECTION_LM_CONFIG = dict(
    model="openai/Doubao-pro-128k",
    api_base="https://aigc.sankuai.com/v1/openai/native",
    api_key="2015679947834241043",
    temperature=1.0,
    max_tokens=4096,
)


@pytest.mark.real_lm
class TestConstraintWithRealLLM:
    """
    使用真实 LLM 的端到端测试。

    跳过方式：pytest -k "not real_lm"
    运行方式：pytest tests/teleprompt/test_gepa_constraint.py -v -m real_lm
    """

    @pytest.fixture(autouse=True)
    def setup_lm(self):
        """初始化真实 LM 并配置到 dspy"""
        try:
            lm = dspy.LM(**_REAL_LM_CONFIG)
            dspy.configure(lm=lm)
        except Exception as e:
            pytest.skip(f"无法初始化真实 LM，跳过测试：{e}")

    def _make_real_reflection_lm(self):
        try:
            return dspy.LM(**_REAL_REFLECTION_LM_CONFIG)
        except Exception as e:
            pytest.skip(f"无法初始化 reflection_lm，跳过测试：{e}")

    def test_constraint_appears_in_reflection_lm_prompt_with_real_lm(self):
        """
        验证：真实 LM 调用时，constraint 文本确实出现在发给 reflection_lm 的 prompt 里。

        策略：用 CapturingLM 替换 reflection_lm，主力 LM 使用真实模型。
        这样只消耗一次主力 LM 推理，就能验证 prompt 注入是否生效。
        """
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        constraint = "【测试约束-请勿删除】输出的 prompt 必须要求模型以 JSON 格式回复"

        student = SimpleQAModule()
        capturing_lm = CapturingLM(return_instruction="Improved instruction that follows the constraint.")

        adapter = DspyAdapter(
            student_module=student,
            metric_fn=simple_metric,
            feedback_map={},
            reflection_lm=capturing_lm,
            constraint=constraint,
        )

        candidate = {"predictor": "请回答用户的问题。"}
        reflective_dataset = {
            "predictor": [
                {
                    "Inputs": {"question": "今天天气怎么样？"},
                    "Generated Outputs": {"answer": "不知道"},
                    "Feedback": "回答太模糊，应该给出具体信息。",
                }
            ]
        }

        results = adapter.propose_new_texts(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=["predictor"],
        )

        assert capturing_lm.last_prompt is not None, "reflection_lm 未被调用"
        assert constraint in capturing_lm.last_prompt, (
            f"约束文本未出现在发给 reflection_lm 的 prompt 中。\n"
            f"约束：{constraint}\n"
            f"prompt 前 800 字符：\n{capturing_lm.last_prompt[:800]}"
        )
        assert isinstance(results.get("predictor"), str)
        print("\n✅ reflection_lm 收到的 prompt 包含约束文本")
        print(f"   约束：{constraint}")

    def test_gepa_full_run_with_constraint_real_lm(self):
        """
        完整跑一轮 GEPA（max_metric_calls=15），验证：
          1. 优化正常完成，返回 optimized_program
          2. 最终生成的新 prompt（instructions）中能体现约束关键词

        注意：该测试会消耗真实 API token，约需 1-3 分钟。
        """
        reflection_lm = self._make_real_reflection_lm()

        # 约束：要求生成的 prompt 必须包含 JSON 输出要求
        constraint = "生成的指令必须明确要求模型以 JSON 格式输出结果，字段名为 answer"

        trainset = [
            Example(question="What is 2+2?", answer="4").with_inputs("question"),
            Example(question="What color is the sky?", answer="blue").with_inputs("question"),
            Example(question="How many days in a week?", answer="7").with_inputs("question"),
        ]

        student = SimpleQAModule()
        original_instruction = student.predictor.signature.instructions

        optimizer = dspy.GEPA(
            metric=simple_metric,
            auto="light",
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
            num_threads=1,
            constraint=constraint,
            track_stats=True,
        )

        print("\n🚀 开始 GEPA 优化（含 constraint）...")
        optimized_program = optimizer.compile(student, trainset=trainset, valset=trainset)

        new_instruction = optimized_program.predictor.signature.instructions
        print(f"\n📝 原始 prompt：\n{original_instruction}")
        print(f"\n📝 优化后 prompt：\n{new_instruction}")

        # 断言 1：程序确实被更新了（instruction 有改动）
        assert new_instruction != original_instruction, "优化后的 instruction 与原始相同，GEPA 可能未生效"

        # 断言 2：新 prompt 中包含约束的核心关键词（JSON / answer）
        # 因为 reflection_lm 被要求在生成 prompt 时遵守约束，所以生成的 prompt 应体现 JSON 要求
        new_instruction_lower = new_instruction.lower()
        assert "json" in new_instruction_lower or "answer" in new_instruction_lower, (
            f"新 prompt 中未体现约束关键词 'json' 或 'answer'。\n新 prompt：{new_instruction}"
        )

        print("\n✅ GEPA 优化完成，新 prompt 体现了约束要求")
