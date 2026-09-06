<p align="center">
  <img align="center" src="docs/docs/static/img/dspy_logo.png" width="460px" />
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## DSPy：以“编程”而非“写 Prompt”的方式驾驭基础模型

**官方文档：** [DSPy 官方技术文档 (dspy.ai)](https://dspy.ai/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)

----

**DSPy** 是专为**系统性“编程（Programming）”而非手工“编写提示词（Prompting）”语言模型**打造的突破性框架。无论您是在构建简单的文本分类器、复杂的 RAG（检索增强生成）流水线，还是自主智能体决策循环（Agent Loops），DSPy 都能帮助您快速**搭建模块化 AI 系统**，并提供先进算法**全自动优化 Prompt 提示词与微调模型权重**。

DSPy 的全称是 **Declarative Self-improving Python（声明式自演进 Python）**。告别脆弱易碎、难以维护的手工提示词工程，您只需编写结构清晰、高度可组合的 *Python 代码*，并借助 DSPy 的编译优化算法**教会您的语言模型持续稳定输出高质量结果**。

欢迎访问我们的 [官方文档站点 (dspy.ai)](https://dspy.ai/) 深入学习，或加入 GitHub 讨论区与 [官方 Discord 社区](https://discord.gg/XCGy2WDCQB) 交流提问与参与贡献。

---

## 📚 官方文档：[dspy.ai](https://dspy.ai)

**完整教程、API 手册与实战范例请查阅：[dspy.ai 官方文档](https://dspy.ai)**

---

## 📦 安装指南 (Installation)

通过 pip 直接安装最新稳定版本：

```bash
pip install dspy
```

如需安装 `main` 分支的最新开发版本：

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
```

---

## 🧩 核心概念全览 (Core Concepts)

DSPy 将传统杂乱的手工 Prompt 转化为结构化、可测试、可自动优化的软件系统：

1. **签名 (Signatures)**：声明输入与输出契约（例如 `question -> answer`），将“任务意图”与“具体 Prompt 措辞”彻底解耦。
2. **模块 (Modules)**：内置 `dspy.Predict`、`dspy.ChainOfThought`、`dspy.ReAct`、`dspy.ProgramOfThought` 等可自由组合的高级推理原语。
3. **优化器 (Optimizers / Teleprompters)**：提供 MIPROv2、BootstrapFewShot、SIMPRO、GEPA 等编译算法，根据评估指标（Metric）自动搜索最优 Few-shot 示例组合与最佳 Instruction。
4. **断言与约束 (Assertions)**：通过计算级约束实现自我修正（Self-refining），确保生成内容百分之百符合业务校验逻辑。

---

## 📜 论文引用与学术阅读 (Citation & Reading More)

如果您希望深入理解 DSPy 背后的理论与前沿演进，推荐研读以下官方研究论文：

* **[2025年7月] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**
* **[2024年6月] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**
* **[2023年10月] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**
* **[2024年7月]** [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)
* **[2024年6月]** [Prompts as Auto-Optimized Training Hyperparameters](https://arxiv.org/abs/2406.11706)
* **[2024年2月]** [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
* **[2024年1月]** [In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178)
* **[2023年12月]** [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)
* **[2022年12月]** [Demonstrate-Search-Predict: Composing Retrieval & Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024.pdf)

实时追踪最新动态，欢迎关注 Twitter 账号 [@DSPyOSS](https://twitter.com/DSPyOSS) 或领英 LinkedIn DSPy 主页。

**DSPy** 官方 Logo 由 **Chuyi Zhang** 设计。

如果您在学术科研或工业应用中使用了 DSPy 或 DSP，请引用以下 BibTeX 条目：

```bibtex
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月6日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
