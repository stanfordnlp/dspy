# 🌟👋 Welcome to DSPy -- The framework for programming—not prompting—foundation models 🌐🚀

<!DOCTYPE html>
<html>
<head>
    <style>
        /* Define the animation */
        @keyframes bounceRotate {
            0% {
                transform: rotateY(0deg);
            }
            50% {
                transform: rotateY(20deg);
            }
            100% {
                transform: rotateY(0deg);
            }
        }

        /* Apply the animation to the image */
        .bounce-rotate-logo {
            animation: bounceRotate 7s ease-in-out infinite; /* Bounce back and forth infinitely with a duration of 5 seconds */
        }
    </style>

</head>
<body>

<p style="text-align:center;">
  <img class="bounce-rotate-logo" src="./docs/images/DSPy8.png" width="50%">
</p>

</body>
</html>

## 🎯 The Vision Behind DSPy

**DSPy** is a framework for developing **high-quality systems** with LMs. While prompting LMs can quickly build (brittle) demos, the best LM systems generally break down problems into steps and tune the prompts or LM weights of each step well. As a bonus, these systems use small LMs to save costs.

This is hard as we usually don't have data to tune each of these steps. **DSPy** treats prompts and LM weights as parameters to be optimized in LM pipelines, given the metrics you want to maximize.

To make this possible:

- [x] **DSPy** provides **composable and declarative modules** for instructing LMs in a familiar Pythonic syntax. It upgrades "prompting techniques" like chain-of-thought and self-reflection from hand-adapted _string manipulation tricks_ into truly modular _generalized operations that learn to adapt to your task_.

- [x] **DSPy** introduces an **automatic compiler that teaches LMs** how to conduct the declarative steps in your program. Specifically, the **DSPy compiler** will internally _trace_ your program and then **craft high-quality prompts for large LMs (or train automatic finetunes for small LMs)** to teach them the steps of your task.

- [x] **DSPy** has many modules and optimizers built-in and we want you to add more. Think of this like PyTorch but for LM pipelines, not DNNs. The **DSPy compiler** _bootstraps_ prompts and finetunes from minimal data **without needing manual labels for the intermediate steps** in your program. Instead of brittle "prompt engineering" with hacky string manipulation, you can explore a systematic space of modular and trainable pieces.

- [x] For complex tasks, **DSPy** can routinely teach powerful models like `GPT-3.5` and local models like `T5-base` or `Llama2-13b` to be much more reliable at tasks. **DSPy** will compile the _same program_ into different few-shot prompts and/or finetunes for each LM.

## 🚀 Analogy to Neural Networks

When we build neural networks, we don't write manual _for-loops_ over lists of _hand-tuned_ floats. Instead, you might use a framework like [PyTorch](https://pytorch.org/) to compose declarative layers (e.g., `Convolution` or `Dropout`) and then use optimizers (e.g., SGD or Adam) to learn the parameters of the network.

Ditto! **DSPy** gives you the right general-purpose modules (e.g., `ChainOfThought`, `Retrieve`, etc.) and takes care of optimizing their prompts _for your program_ and your metric, whatever they aim to do. Whenever you modify your code, your data, or your validation constraints, you can _compile_ your program again and **DSPy** will create new effective prompts that fit your changes.

**Welcome to the future of LLMs programmig! 🌟🌐**
