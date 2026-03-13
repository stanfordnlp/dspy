"""
This is an implementation of the RP-ReAct agent architecture using DSPy ReAct module (dspy.ReAct).

The RP-ReAct approach splits complex agent work into two roles to reduce cognitive overload and improve reliability.
A Reasoner-Planner Agent (RPA) handles high-level strategy, breaking a user request into simple sub-questions and
adapting the plan based on results.
A Proxy-Execution Agent (PEA) handles low-level execution, translating each sub-question into precise tool calls with
correct parameters and syntax.

RP-ReAct Methodology adapted to DSPy from the original paper (Giargiari et al. 2025) https://arxiv.org/pdf/2512.03560
Proof Code from the paper — https://github.com/giargiapower/RP-ReAct
"""
import dspy
from dspy.signatures.signature import Signature
from dspy.predict.react import ReAct
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
import inspect
from typing import Any, Callable


class RPReAct(Module):
	def __init__(
			self,
			signature: type["Signature"],
			tools: list[Callable],
			max_iters: int = 20,
			planner_lm: dspy.LM = None,
			executor_lm: dspy.LM = None,
	):
		"""
		The RP-ReAct approach splits complex agent work into two roles to reduce cognitive overload and improve
		reliability.
		
		* A Reasoner-Planner Agent (RPA) handles high-level strategy, breaking a user request into simple sub-questions
		and adapting the plan based on results.
		* A Proxy-Execution Agent (PEA) handles low-level execution, translating each sub-question into precise tool
		calls with correct parameters and syntax.

		Args:
			signature: The signature of the overall RP-ReAct module.
			tools: A list of callables that the executor can use to perform specific tasks.
			max_iters: The maximum number of iterations for both the planner and executor to prevent infinite loops.
			planner_lm: The language model to use for the planner. If None, default LM configured in dspy will be used.
			executor_lm: The language model to use for the executor. If None, default LM configured in dspy will be used.

		Example usage:

		```python
		def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."

        rpreact = dspy.RPReAct(signature="question->answer", tools=[get_weather])
        pred = rpreact(question="What is the weather in Tokyo?")
		```
		
		"""
		super().__init__()
		self.signature = signature = ensure_signature(signature)
		self.tools = tools
		self.max_iters = max_iters
		self.planner_lm = planner_lm
		self.executor_lm = executor_lm
		
		self.react_executor = ReAct(signature, tools=self.tools, max_iters=max_iters)

		def executor_tool(**kwargs: Any) -> Any:
			dspy.configure(lm=self.executor_lm)
			with dspy.context(lm=self.executor_lm):
				result = self.react_executor(**kwargs)
			
			# Restore planner's LM
			dspy.configure(lm=self.planner_lm)

			result = {k: getattr(result, k) for k in signature.output_fields.keys()}
			return result

		executor_tool.__signature__ = inspect.Signature(
			parameters=[
				inspect.Parameter(name=k, kind=inspect.Parameter.KEYWORD_ONLY)
				for k in signature.input_fields.keys()
			],
			return_annotation=signature.output_fields,
		)

		planner_instructions = [f"{signature.instructions}\n"] if signature.instructions else []

		inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
		outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])

		planner_instructions.extend(
            [
				"Plan the next sub-question or provide a final answer.",
				"- Divide the main task/question into simple and non-nested sub-questions.",
				f"- Use the tool `{executor_tool.__name__}` to execute sub-steps as needed. Inputs: {inputs}. Outputs: {outputs}.",
				"- If you are done, provide a final answer."
			]
		)
	
		planner_signature = signature.with_instructions("\n".join(planner_instructions))
		self.react_planner = ReAct(planner_signature, tools=[executor_tool], max_iters=max_iters)

	def forward(self, **kwargs: Any) -> Any:
		dspy.configure(lm=self.planner_lm)
		with dspy.context(lm=self.planner_lm):
			result = self.react_planner(**kwargs)
		return result
