---
draft: true
---

!!! warning "This document is from Aug 2024. Since then, DSPy 2.5 and 2.6 were released, DSPy has grown considerably, and 3.0 is approaching! Content below is highly outdated."



# Roadmap Sketch: DSPy 2.5+

It’s been a year since DSPy evolved out of Demonstrate–Search–Predict (DSP), whose research started at Stanford NLP all the way back in February 2022. Thanks to 200 wonderful contributors, DSPy has introduced tens of thousands of people to building modular LM programs and optimizing their prompts and weights automatically. In this time, DSPy has grown to 160,000 monthly downloads and 16,000 stars on GitHub, becoming synonymous with prompt optimization in many circles and inspiring at least a half-dozen cool new libraries.

This document is an initial sketch of DSPy’s public roadmap for the next few weeks and months, as we work on DSPy 2.5 and plan for DSPy 3.0. Suggestions and open-source contributors are more than welcome: just open an issue or submit a pull request regarding the roadmap.



## Technical Objectives

The thesis of DSPy is that for LMs to be useful, we have to shift from ad-hoc prompting to new notions of programming LMs. Instead of relying on LMs gaining much more general or more compositional capabilities, we need to enable developers to iteratively explore their problems and build modular software that invokes LMs for well-scoped tasks. We need to enable that through modules and optimizers that isolate how they decompose their problems and describe their system's objectives from how their LMs are invoked or fine-tuned to maximize their objectives. DSPy's goal has been to develop (and to build the community and shared infrastructure for the collective development of) the abstractions, programming patterns, and optimizers toward this thesis.

To a first approximation, DSPy’s current user-facing language has the minimum number of appropriate abstractions that address the goals above: declarative signatures, define-by-run modules, and optimizers that can be composed quite powerfully. But there are several things we need to do better to realize our goals. The upcoming DSPy releases will have the following objectives.

1. Polishing the core functionality.
2. Developing more accurate, lower-cost optimizers.
3. Building end-to-end tutorials from DSPy’s ML workflow to deployment.
4. Shifting towards more interactive optimization & tracking.



## Team & Organization

DSPy is fairly unusual in its technical objectives, contributors, and audience. Though DSPy takes inspiration from PyTorch, a library for building and optimizing DNNs, there is one major difference: PyTorch was introduced well after DNNs were mature ML concepts, but DSPy seeks to establish and advance core LM Programs research: the framework is propelled by constant academic research from programming abstractions (like the original **Demonstrate–Search–Predict** concepts, DSPy **Signatures**, or **LM Assertions**) to NLP systems (like **STORM**, **PATH**, and **IReRa**) to prompt optimizers (like **MIPRO**) and RL (like **BetterTogether**), among many other related directions.

This research all composes into a concrete, practical library, thanks to dozens of industry contributors, many of whom are deploying apps in production using DSPy. Because of this, DSPy reaches not only of grad students and ML engineers, but also many non-ML engineers, from early adopter SWEs to hobbyists exploring new ways of using LMs. The following team, with help from many folks in the OSS community, is working towards the objectives in this Roadmap.

**Project Lead:** Omar Khattab (Stanford & Databricks)

**Project Mentors:** Chris Potts (Stanford), Matei Zaharia (UC Berkeley & Databricks), Heather Miller (CMU & Two Sigma)

**Core Library:** Arnav Singhvi (Databricks & Stanford), Herumb Shandilya (Stanford), Hanna Moazam (Databricks), Sri Vardhamanan (Dashworks), Cyrus Nouroozi (Zenbase), Amir Mehr (Zenbase), Kyle Caverly (Modular), with special thanks to Keshav Santhanam (Stanford), Thomas Ahle (Normal Computing), Connor Shorten (Weaviate)

**Prompt Optimization:** Krista Opsahl-Ong (Stanford), Michael Ryan (Stanford), Josh Purtell (Basis), with special thanks to Eric Zhang (Stanford)

**Finetuning & RL:** Dilara Soylu (Stanford), Isaac Miller (Anyscale), Karel D'Oosterlinck (Ghent), with special thanks to Paridhi Masehswari (Stanford)

**PL Abstractions:** Shangyin Tan (UC Berkeley), Manish Shetty (UC Berkeley), Peter Zhong (CMU)

**Applications:** Jasper Xian (Waterloo), Saron Samuel (Stanford), Alberto Mancarella (Stanford), Faraz Khoubsirat (Waterloo), Saiful Haq (IIT-B), Ashutosh Sharma (UIUC)



## 1) Polishing the core functionality.

Over the next month, polishing is the main objective and likely the one to have the highest ROI on the experience of the average user. Conceptually, DSPy has an extremely small core. It’s nothing but (1) LMs, (2) Signatures & Modules, (3) Optimizers, and (4) Assertions. These concepts and their implementations evolved organically over the past couple of years. We are working now to consolidate what we’ve learned and refactor internally so that things “just work” out of the box for new users, who may not know all the tips-and-tricks just yet.

More concretely:

1. We want to increase the quality of zero-shot, off-the-shelf DSPy programs, i.e. those not yet compiled on custom data.
2. Wherever possible, DSPy should delegate lower-level internal complexity (like managing LMs and structured generation) to emerging lower-level libraries. When required, we may fork smaller libraries out of DSPy to support infrastructure pieces as their own projects.
3. DSPy should internally be more modular and we need higher compatibility between internal components. Specifically, we need more deeper and more native investment in (i) typed multi-field constraints, (ii) assertions, (iii) observability and experimental tracking, (iv) deployment of artifacts and related concerns like streaming and async, and (v) fine-tuning and serving open models.


### On LMs

As of DSPy 2.4, the library has approximately 20,000 lines of code and roughly another 10,000 lines of code for tests, examples, and documentation. Some of these are clearly necessary (e.g., DSPy optimizers) but others exist only because the LM space lacks the building blocks we need under the hood. Luckily, for LM interfaces, a very strong library now exists: LiteLLM, a library that unifies interfaces to various LM and embedding providers. We expect to reduce around 6000 LoCs of support for custom LMs and retrieval models by shifting a lot of that to LiteLLM.

Objectives in this space include improved caching, saving/loading of LMs, support for streaming and async LM requests. Work here is currently led by Hanna Moazam and Sri Vardhamanan, building on a foundation by Cyrus Nouroozi, Amir Mehr, Kyle Caverly, and others.


### On Signatures & Modules

Traditionally, LMs offer text-in-text-out interfaces. Toward modular programming, DSPy introduced signatures for the first time (as DSP Templates in Jan 2023) as a way to structure the inputs and outputs of LM interactions. Standard prompts conflate interface (“what should the LM do?”) with implementation (“how do we tell it to do that?”). DSPy signatures isolate the former so we can infer and learn the latter from data — in the context of a bigger program. Today in the LM landscape, notions of "structured outputs" have evolved dramatically, thanks to constrained decoding and other improvements, and have become mainstream. What may be called "structured inputs" remains is yet to become mainstream outside of DSPy, but is as crucial.

Objectives in this space include refining the abstractions and implementations first-class notion of LM Adapters in DSPy, as translators that sits between signatures and LM interfaces. While Optimizers adjust prompts through interactions with a user-supplied metric and data, Adapters are more concerned with building up interactions with LMs to account for, e.g. (i) non-plaintext LM interfaces like chat APIs, structured outputs, function calling, and multi-modal APIs, (ii) languages beyond English or other forms of higher-level specialization. This has been explored in DSPy on and off in various forms, but we have started working on more fundamental approaches to this problem that will offer tangible improvements to most use-cases. Work here is currently led by Omar Khattab.


### On Finetuning & Serving

In February 2023, DSPy introduced the notion of compiling to optimize the weights of an LM program. (To understand just how long ago that was in AI terms, this was before the Alpaca training project at Stanford had even started and a month before the first GPT-4 was released.) Since then, we have shown in October 2023 and, much more expansively, in July 2024, that the fine-tuning flavor of DSPy can deliver large gains for small LMs, especially when composed with prompt optimization.

Overall, though, most DSPy users in practice explore prompt optimization and not weight optimization and most of our examples do the same. The primary reason for a lot of this is infrastructure. Fine-tuning in the DSPy flavor is more than just training a model: ultimately, we need to bootstrap training data for several different modules in a program, train multiple models and handle model selection, and then load and plug in those models into the program's modules. Doing this robustly at the level of abstraction DSPy offers requires a level of resource management that is not generally supported by external existing tools. Major efforts in this regard are currently led by Dilara Soylu and Isaac Miller.


### On Optimizers & Assertions

This is a naturally major direction in the course of polishing. We will share more thoughts here after making more progress on the three angles above.



## 2) Developing more accurate, lower-cost optimizers.

A very large fraction of the research in DSPy focuses on optimizing the prompts and the weights of LM programs. In December 2022, we introduced the algorithm and abstractions behind BootstrapFewShot (as Demonstrate in DSP) and several of its variants. In February 2023, we introduced the core version of what later became BootstrapFinetune. In August 2023, we introduced new variations of both of these. In December 2023, we introduced the first couple of instruction optimizers into DSPy, CA-OPRO and early versions of MIPRO. These were again upgraded in March 2024. Fast forward to June and July 2024, we released MIPROv2 for prompt optimization and BetterTogether for fine-tuning the weights of LM programs.

We have been working towards a number of stronger optimizers. While we cannot share the internal details of research on new optimizers yet, we can outline the goals. A DSPy optimizer can be characterized via three angles:

1. Quality: How much quality can it deliver from various LMs? How effective does it need the zero-shot program to be in order to work well?
2. Cost: How many labeled (and unlabeled) inputs does it need? How many invocations of the program does it need? How expensive is the resulting optimized program at inference time?
3. Robustness: How well can it generalize to different unseen data points or distributions? How sensitive is it to mistakes of the metric or labels?

Over the next six months, our goal is to dramatically improve each angle of these _when the other two are held constant_.  Concretely, there are three directions here.

- Benchmarking: A key prerequisite here is work on benchmarking. On the team, Michael Ryan and Shangyin Tan are leading these efforts. More soon.

- Quality: The goal here is optimizers that extract, on average, 20% more on representative tasks than MIPROv2 and BetterTogether, under the usual conditions — like a few hundred inputs with labels and a good metric starting from a decent zero-shot program. Various efforts here are led by Dilara Soylu, Michael Ryan, Josh Purtell, Krista Opsahl-Ong, and Isaac Miller.

- Efficiency: The goal here is optimizers that match the current best scores from MIPROv2 and BetterTogether but under 1-2 challenges like: (i) starting from only 10-20 inputs with labels, (ii) starting with a weak zero-shot program that scores 0%, (iii) where significant misalignment exists between train/validation and test, or (iii) where the user supplies no metric but provides a very small number of output judgments.



## 3) Building end-to-end tutorials from DSPy’s ML workflow to deployment.

Using DSPy well for solving a new task is just doing good machine learning with LMs, but teaching this is hard. On the one hand, it's an iterative process: you make some initial choices, which will be sub-optimal, and then you refine them incrementally. It's highly exploratory: it's often the case that no one knows yet how to best solve a problem in a DSPy-esque way. One the other hand, DSPy offers many emerging lessons from several years of building LM systems, in which the design space, the data regime, and many other factors are new both to ML experts and to the very large fraction of users that have no ML experience.

Though current docs do address [a bunch of this](learn/index.md) in isolated ways, one thing we've learned is that we should separate teaching the core DSPy language (which is ultimately pretty small) from teaching the emerging ML workflow that works well in a DSPy-esque setting. As a natural extension of this, we need to place more emphasis on steps prior and after to the explicit coding in DSPy, from data collection to deployment that serves and monitors the optimized DSPy program in practice. This is just starting but efforts will be ramping up led by Omar Khattab, Isaac Miller, and Herumb Shandilya.


## 4) Shifting towards more interactive optimization & tracking.

Right now, a DSPy user has a few ways to observe and tweak the process of optimization. They can study the prompts before, during, and after optimization methods like `inspect_history`, built-in logging, and/or the metadata returned by optimizers. Similarly, they can rely on `program.save` and `program.load` to potentially adjust the optimized prompts by hand. Alternatively, they can use one of the many powerful observability integrations — like from Phoenix Arize, LangWatch, or Weights & Biases Weave — to observe _in real time_ the process of optimization (e.g., scores, stack traces, successful & failed traces, and candidate prompts). DSPy encourages iterative engineering by adjusting the program, data, or metrics across optimization runs. For example, some optimizers allow “checkpointing” — e.g., if you optimize with BootstrapFewShotWithRandomSearch for 10 iterations then increase to 15 iterations, the first 10 will be loaded from cache.

While these can accomplish a lot of goals, there are two limitations that future versions of DSPy will seek to address.

1. In general, DSPy’s (i) observability, (ii) experimental tracking, (iii) cost management, and (iii) deployment of programs should become first-class concerns via integration with tools like MLFlow. We will share more plans addressing this for DSPy 2.6 in the next 1-2 months.

2. DSPy 3.0 will introduce new optimizers that prioritize ad-hoc, human-in-the-loop feedback. This is perhaps the only substantial paradigm shift we see as necessary in the foreseeable future in DSPy. It involves various research questions at the level of the abstractions, UI/HCI, and ML, so it is a longer-term goal that we will share more about in the next 3-4 month.


