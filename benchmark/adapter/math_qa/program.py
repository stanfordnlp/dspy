import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import pydantic
from tqdm import tqdm

import dspy

from .metric import is_equiv, is_equiv_dspy


class MathQA(dspy.Signature):
    """You are a math expert, and you will be given a math problem and you will need to solve it."""

    question: str = dspy.InputField(desc="Math question to answer")
    answer: str = dspy.OutputField()


def _dspy_cot(dataset, model, adapter, num_threads=1):
    adapter_name = adapter.__class__.__name__
    print(f"""
========================================
         Math Question Answering
========================================
- Runs with:  DSPy Program
- Adapter:    {adapter_name}
- Model:      {model}
----------------------------------------
Running benchmarking, this may take a while...
""")
    cot = dspy.ChainOfThought(MathQA)

    evaluator = dspy.Evaluate(devset=dataset, num_threads=num_threads, display_progress=True, display_table=False)

    with dspy.settings.context(lm=dspy.LM(model), adapter=adapter):
        start_time = time.time()
        score = evaluator(cot, metric=is_equiv_dspy)
        time_taken = time.time() - start_time

        print(f"""
- Score:      {score}
- Time taken: {time_taken:.2f} seconds
========================================
""")

    return {"score": score}


def dspy_cot_chat_adapter(dataset, model, num_threads=1):
    return _dspy_cot(dataset, model, dspy.ChatAdapter(), num_threads)


def dspy_cot_json_adapter(dataset, model, num_threads=1):
    return _dspy_cot(dataset, model, dspy.JSONAdapter(), num_threads)


def _vanilla_sdk(dataset, model, num_threads=1, lm_kwargs=None):
    prompt = """You are a math expert. Solve the following math problem step by step, and provide the final answer.
    Please make sure the answer only contains the final answer, no other text like reasoning steps or explanation should
    be included. for example: if the answer is 10, the response should be "10" instead of "The answer is 10".

Question: {question}
Answer:"""

    scores = []

    start_time = time.time()
    lm_kwargs = lm_kwargs or {}

    print(f"""
========================================
         Math Question Answering
========================================
- Runs with:  Vanilla LM SDK
- Model:      {model}
- Using structured output: {"response_format" in lm_kwargs}
----------------------------------------
Running benchmarking, this may take a while...
""")

    def process_example(example):
        lm_kwargs["caching"] = True
        response = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt.format(question=example.question)}], **lm_kwargs
        )
        if "response_format" in lm_kwargs:
            pred_answer = json.loads(response["choices"][0].message.content).get("answer", "")
        else:
            pred_answer = response["choices"][0].message.content

        return is_equiv(example.answer, pred_answer)

    scores = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(process_example, example) for example in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="ex"):
            scores.append(future.result())

    time_taken = time.time() - start_time

    score = sum(scores) * 100.0 / len(scores)

    print(f"""
- Score:      {score}
- Time taken: {time_taken:.2f} seconds
========================================
""")
    return {"score": score}


def vanilla_sdk(dataset, model, num_threads=1):
    return _vanilla_sdk(dataset, model, num_threads)


def vanilla_sdk_with_structured_output(dataset, model, num_threads=1):
    lm_kwargs = {
        "response_format": {
            "type": "json_object",
        }
    }

    class MathAnswer(pydantic.BaseModel):
        answer: str

    lm_kwargs["response_format"] = MathAnswer

    return _vanilla_sdk(dataset, model, num_threads, lm_kwargs)
