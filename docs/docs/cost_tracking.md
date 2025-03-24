# Cost Tracking in DSPy

DSPy provides built-in cost tracking capabilities that allow you to monitor and analyze the costs of your language model calls. This guide explains how to access and use this functionality.

## Basic Usage

The cost tracking information is automatically available in the metadata of any prediction returned by DSPy modules. Here's a simple example:

```python
import dspy

# Create a simple predictor
predictor = dspy.Predict("question -> answer")

# Make a prediction
result = predictor(question="What is the capital of France?")

# Access cost information
print(f"Cost: ${result.metadata.cost}")
print(f"Token Usage: {result.metadata.usage}")
print(f"Model Used: {result.metadata.model}")
print(f"Timestamp: {result.metadata.timestamp}")
print(f"Trace ID: {result.metadata.trace_id}")
```

## Cost Tracking in ReAct

When using ReAct, you can track costs for each reasoning step:

```python
import dspy
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReActWithCostTracking(dspy.ReAct):
    def __init__(self, signature, tools: list[callable], max_iters=5):
        super().__init__(signature, tools, max_iters)
        self.total_cost = 0
        self.step_costs = []
        self.total_tokens = 0
        self.step_tokens = []

    def forward(self, **input_args):
        trajectory = {}
        self.total_cost = 0
        self.step_costs = []
        self.total_tokens = 0
        self.step_tokens = []

        for idx in range(self.max_iters):
            logger.info(f"Starting ReAct iteration {idx + 1}")
            
            # Get prediction for this step
            pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            
            # Track costs and tokens for this step
            if pred.metadata:
                step_cost = pred.metadata["cost"] or 0
                step_tokens = pred.metadata["usage"].get("total_tokens", 0)
                
                self.total_cost += step_cost
                self.total_tokens += step_tokens
                self.step_costs.append(step_cost)
                self.step_tokens.append(step_tokens)
                
                logger.info(f"Step {idx + 1} - Cost: ${step_cost:.4f}, Tokens: {step_tokens}")

            # Update trajectory with step information
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                # Execute the tool
                tool = self.tools[pred.next_tool_name]
                parsed_tool_args = {}
                for k, v in pred.next_tool_args.items():
                    if hasattr(tool, "arg_types") and k in tool.arg_types:
                        arg_type = tool.arg_types[k]
                        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
                            parsed_tool_args[k] = arg_type.model_validate(v)
                            continue
                    parsed_tool_args[k] = v
                
                trajectory[f"observation_{idx}"] = tool(**parsed_tool_args)
                logger.info(f"Tool {pred.next_tool_name} executed successfully")
                
            except Exception as e:
                trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"
                logger.error(f"Error executing tool {pred.next_tool_name}: {e}")

            # Check if we're done
            if pred.next_tool_name == "finish":
                logger.info("ReAct process completed successfully")
                break

        # Get final answer
        logger.info("Getting final answer")
        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        
        # Add cost information to the final prediction
        return dspy.Prediction(
            trajectory=trajectory,
            total_cost=self.total_cost,
            step_costs=self.step_costs,
            total_tokens=self.total_tokens,
            step_tokens=self.step_tokens,
            **extract
        )

# Example usage with a simple calculator tool
def calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator tool that can perform basic arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    return operations.get(operation, lambda x, y: "Error: Invalid operation")(a, b)

# Create a ReAct instance with cost tracking
react = ReActWithCostTracking(
    signature="question -> answer",
    tools=[calculator],
    max_iters=5
)

# Example usage
result = react(question="""
    Let's solve this step by step:
    1. First, multiply 5 and 3
    2. Then add 10 to the result
    3. Finally, divide by 2
    What's the final answer?
""")

# Print detailed results
print("\n=== ReAct Results ===")
print(f"Final Answer: {result.answer}")
print("\n=== Reasoning Process ===")
for i in range(len(result.trajectory) // 4):  # Each step has 4 entries
    print(f"\nStep {i + 1}:")
    print(f"Thought: {result.trajectory[f'thought_{i}']}")
    print(f"Tool: {result.trajectory[f'tool_name_{i}']}")
    print(f"Arguments: {result.trajectory[f'tool_args_{i}']}")
    print(f"Observation: {result.trajectory[f'observation_{i}']}")

print("\n=== Cost Breakdown ===")
print(f"Total Cost: ${result.total_cost:.4f}")
print(f"Total Tokens: {result.total_tokens}")
print("\nPer Step:")
for i, (cost, tokens) in enumerate(zip(result.step_costs, result.step_tokens)):
    print(f"Step {i + 1}:")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Tokens: {tokens}")
```

## Using Callbacks for Cost Tracking

You can also use DSPy's callback system to track costs across your entire application:

```python
import dspy
from dspy.utils.callback import BaseCallback

class CostTrackingCallback(BaseCallback):
    def __init__(self):
        self.total_cost = 0
        self.total_tokens = 0

    def on_lm_end(self, call_id, outputs, exception):
        if outputs and "metadata" in outputs:
            metadata = outputs["metadata"]
            if metadata["cost"]:
                self.total_cost += metadata["cost"]
            if metadata["usage"]:
                self.total_tokens += metadata["usage"].get("total_tokens", 0)

    def get_summary(self):
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens
        }

# Configure DSPy to use the callback
callback = CostTrackingCallback()
dspy.settings.configure(callbacks=[callback])

# Use DSPy as normal
predictor = dspy.Predict("question -> answer")
result = predictor(question="What is the capital of France?")

# Get cost summary
summary = callback.get_summary()
print(f"Total cost: ${summary['total_cost']}")
print(f"Total tokens: {summary['total_tokens']}")
```

## Cost Tracking in Chain of Thought

When using Chain of Thought, you can track costs for both the reasoning and final answer steps:

```python
import dspy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoTWithCostTracking(dspy.ChainOfThought):
    def __init__(self, signature, **kwargs):
        super().__init__(signature, **kwargs)
        # Initialize the reasoning and answer predictors
        self.reasoning = dspy.Predict("question -> reasoning")
        self.answer = dspy.Predict("question, reasoning -> answer")
        logger.info("Initialized CoTWithCostTracking")

    def forward(self, **kwargs):
        try:
            # Step 1: Get reasoning
            logger.info("Starting reasoning step")
            reasoning_pred = self.reasoning(**kwargs)
            reasoning_cost = reasoning_pred.metadata["cost"]
            logger.info(f"Reasoning step completed. Cost: ${reasoning_cost}")
            
            # Step 2: Get answer
            logger.info("Starting answer step")
            answer_pred = self.answer(
                question=kwargs["question"],
                reasoning=reasoning_pred.reasoning
            )
            answer_cost = answer_pred.metadata["cost"]
            logger.info(f"Answer step completed. Cost: ${answer_cost}")
            
            # Calculate total cost
            total_cost = (reasoning_cost or 0) + (answer_cost or 0)
            logger.info(f"Total cost: ${total_cost}")
            
            return dspy.Prediction(
                reasoning=reasoning_pred.reasoning,
                answer=answer_pred.answer,
                total_cost=total_cost,
                reasoning_cost=reasoning_cost,
                answer_cost=answer_cost,
                reasoning_tokens=reasoning_pred.metadata["usage"].get("total_tokens", 0),
                answer_tokens=answer_pred.metadata["usage"].get("total_tokens", 0)
            )
            
        except Exception as e:
            logger.error(f"Error in CoT process: {e}")
            raise

# Usage example
cot = CoTWithCostTracking("question -> reasoning, answer")

# Example with a complex question
result = cot(question="""
    A farmer has 17 sheep, and all but 9 die.
    How many sheep are left?
    Please show your reasoning step by step.
""")

# Print detailed results
print("\n=== Chain of Thought Results ===")
print(f"Reasoning Process:\n{result.reasoning}")
print(f"\nFinal Answer: {result.answer}")
print("\n=== Cost Breakdown ===")
print(f"Reasoning Step:")
print(f"  Cost: ${result.reasoning_cost}")
print(f"  Tokens: {result.reasoning_tokens}")
print(f"\nAnswer Step:")
print(f"  Cost: ${result.answer_cost}")
print(f"  Tokens: {result.answer_tokens}")
print(f"\nTotal:")
print(f"  Cost: ${result.total_cost}")
print(f"  Total Tokens: {result.reasoning_tokens + result.answer_tokens}")
```

## Budget-Aware Execution

DSPy allows you to implement budget-aware execution to control costs. Here's an example of how to create a budget-aware ReAct module:

```python
import dspy
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BudgetAwareReAct(dspy.ReAct):
    def __init__(self, signature, tools: list[callable], max_iters=5, budget=1.0):
        super().__init__(signature, tools, max_iters)
        self.budget = budget
        self.remaining_budget = budget
        self.total_cost = 0
        self.step_costs = []
        self.step_tokens = []

    def forward(self, **input_args):
        trajectory = {}
        self.remaining_budget = self.budget
        self.total_cost = 0
        self.step_costs = []
        self.step_tokens = []

        for idx in range(self.max_iters):
            if self.remaining_budget <= 0:
                logger.warning("Budget exceeded, stopping ReAct process")
                break
                
            logger.info(f"Starting ReAct iteration {idx + 1} (Budget remaining: ${self.remaining_budget:.4f})")
            
            # Get prediction for this step
            pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            
            # Track costs and tokens for this step
            if pred.metadata:
                step_cost = pred.metadata["cost"] or 0
                step_tokens = pred.metadata["usage"].get("total_tokens", 0)
                
                if step_cost > self.remaining_budget:
                    logger.warning(f"Step {idx + 1} would exceed budget (Cost: ${step_cost:.4f}, Remaining: ${self.remaining_budget:.4f})")
                    break
                
                self.remaining_budget -= step_cost
                self.total_cost += step_cost
                self.step_costs.append(step_cost)
                self.step_tokens.append(step_tokens)
                
                logger.info(f"Step {idx + 1} - Cost: ${step_cost:.4f}, Tokens: {step_tokens}, Remaining Budget: ${self.remaining_budget:.4f}")

            # Update trajectory with step information
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                # Execute the tool
                tool = self.tools[pred.next_tool_name]
                parsed_tool_args = {}
                for k, v in pred.next_tool_args.items():
                    if hasattr(tool, "arg_types") and k in tool.arg_types:
                        arg_type = tool.arg_types[k]
                        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
                            parsed_tool_args[k] = arg_type.model_validate(v)
                            continue
                    parsed_tool_args[k] = v
                
                trajectory[f"observation_{idx}"] = tool(**parsed_tool_args)
                logger.info(f"Tool {pred.next_tool_name} executed successfully")
                
            except Exception as e:
                trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"
                logger.error(f"Error executing tool {pred.next_tool_name}: {e}")

            # Check if we're done
            if pred.next_tool_name == "finish":
                logger.info("ReAct process completed successfully")
                break

        # Get final answer if we have budget remaining
        if self.remaining_budget > 0:
            logger.info("Getting final answer")
            extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        else:
            logger.warning("Insufficient budget for final answer")
            extract = {"answer": "Unable to complete due to budget constraints"}
        
        # Add cost information to the final prediction
        return dspy.Prediction(
            trajectory=trajectory,
            total_cost=self.total_cost,
            step_costs=self.step_costs,
            total_tokens=sum(self.step_tokens),
            step_tokens=self.step_tokens,
            remaining_budget=self.remaining_budget,
            budget_exceeded=self.remaining_budget <= 0,
            **extract
        )

# Example usage with a simple calculator tool
def calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator tool that can perform basic arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    return operations.get(operation, lambda x, y: "Error: Invalid operation")(a, b)

# Create a budget-aware ReAct instance
react = BudgetAwareReAct(
    signature="question -> answer",
    tools=[calculator],
    max_iters=5,
    budget=0.5  # Set a budget of $0.50
)

# Example usage
result = react(question="""
    Let's solve this step by step:
    1. First, multiply 5 and 3
    2. Then add 10 to the result
    3. Finally, divide by 2
    What's the final answer?
""")

# Print detailed results
print("\n=== Budget-Aware ReAct Results ===")
print(f"Final Answer: {result.answer}")
print(f"Budget Status: {'Exceeded' if result.budget_exceeded else 'Within Budget'}")
print(f"Remaining Budget: ${result.remaining_budget:.4f}")

print("\n=== Reasoning Process ===")
for i in range(len(result.trajectory) // 4):  # Each step has 4 entries
    print(f"\nStep {i + 1}:")
    print(f"Thought: {result.trajectory[f'thought_{i}']}")
    print(f"Tool: {result.trajectory[f'tool_name_{i}']}")
    print(f"Arguments: {result.trajectory[f'tool_args_{i}']}")
    print(f"Observation: {result.trajectory[f'observation_{i}']}")

print("\n=== Cost Breakdown ===")
print(f"Total Cost: ${result.total_cost:.4f}")
print(f"Total Tokens: {sum(result.step_tokens)}")
print("\nPer Step:")
for i, (cost, tokens) in enumerate(zip(result.step_costs, result.step_tokens)):
    print(f"Step {i + 1}:")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Tokens: {tokens}")
```

This budget-aware implementation:
1. Sets a maximum budget for the entire ReAct process
2. Tracks remaining budget after each step
3. Stops execution if budget would be exceeded
4. Provides detailed cost tracking per step
5. Includes budget status in the final prediction
6. Logs budget-related warnings and information

## Best Practices

1. **Cost Monitoring**: Always check costs in development to ensure your prompts are efficient.
2. **Error Handling**: The cost field may be `None` if the provider doesn't return cost information.
3. **Token Usage**: The `usage` field contains detailed token counts (prompt_tokens, completion_tokens, total_tokens).
4. **Trace IDs**: Use the `trace_id` to correlate costs with specific calls in your logs.
5. **Timestamps**: The `timestamp` field helps track when each call was made.

## Limitations

- Cost tracking depends on the LLM provider's ability to return cost information
- Some providers may not return detailed usage statistics
- Costs are in the provider's native currency (usually USD)
- Token usage may vary between providers due to different tokenization methods

## Example with Multiple Providers

```python
import dspy

# Configure multiple LMs
gpt4 = dspy.LM("openai/gpt-4", your_api_key)
claude = dspy.LM("anthropic/claude-3-opus-20240229", your_api_key)

# Create a predictor that can use either model
predictor = dspy.Predict("question -> answer")

# Track costs for each model
for model in [gpt4, claude]:
    dspy.settings.configure(lm=model)
    result = predictor(question="What is the capital of France?")
    print(f"\nUsing {model.model}:")
    print(f"Cost: ${result.metadata["cost"]}")
    print(f"Tokens: {result.metadata["usage"]}")
```
