---
sidebar_position: 2
---

# [1] DSPy Signatures

## 1) What is a DSPy Signature?

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.**

What does that mean? It means tht signatures allow you tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.


## 2) Why are they called Signatures?

You're probably familiar with function signatures, which specify the input and output arguments and their types.

Ditto. DSPy signatures are similar, but the differences are that:

- While typical function signatures just _describe_ things, DSPy Signatures _define and control the behavior_ of modules.

- The field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.


## 3) Why should I use a DSPy Signature?

**tl;dr** For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes).

**Long Answer:** Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning.

Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

