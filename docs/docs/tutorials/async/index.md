# Async DSPy Programming

DSPy provides native support for asynchronous programming, allowing you to build more efficient and
scalable applications. This guide will walk you through how to leverage async capabilities in DSPy,
covering both built-in modules and custom implementations.

## Why Use Async in DSPy?

Asynchronous programming in DSPy offers several benefits:
- Improved performance through concurrent operations
- Better resource utilization
- Reduced waiting time for I/O-bound operations
- Enhanced scalability for handling multiple requests

## When Should I use Sync or Async?

Choosing between synchronous and asynchronous programming in DSPy depends on your specific use case.
Here's a guide to help you make the right choice:

Use Synchronous Programming When

- You're exploring or prototyping new ideas
- You're conducting research or experiments
- You're building small to medium-sized applications
- You need simpler, more straightforward code
- You want easier debugging and error tracking

Use Asynchronous Programming When:

- You're building a high-throughput service (high QPS)
- You're working with tools that only support async operations
- You need to handle multiple concurrent requests efficiently
- You're building a production service that requires high scalability

### Important Considerations

While async programming offers performance benefits, it comes with some trade-offs:

- More complex error handling and debugging
- Potential for subtle, hard-to-track bugs
- More complex code structure
- Different code between ipython (Colab, Jupyter lab, Databricks notebooks, ...) and normal python runtime.

We recommend starting with synchronous programming for most development scenarios and switching to async
only when you have a clear need for its benefits. This approach allows you to focus on the core logic of
your application before dealing with the additional complexity of async programming.

## Using Built-in Modules Asynchronously

Most DSPy built-in modules support asynchronous operations through the `acall()` method. This method
maintains the same interface as the synchronous `__call__` method but operates asynchronously.

Here's a basic example using `dspy.Predict`:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    # Use acall() for async execution
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)


asyncio.run(main())
```

### Working with Async Tools

DSPy's `Tool` class seamlessly integrates with async functions. When you provide an async
function to `dspy.Tool`, you can execute it using `acall()`. This is particularly useful
for I/O-bound operations or when working with external services.

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    # Simulate an async operation
    await asyncio.sleep(0.1)
    print(f"I get: {x}")

# Create a tool from the async function
tool = dspy.Tool(foo)

async def main():
    # Execute the tool asynchronously
    await tool.acall(x=2)

asyncio.run(main())
```

#### Using Async Tools in Synchronous Contexts

If you need to call an async tool from synchronous code, you can enable automatic async-to-sync conversion:

```python
import dspy

async def async_tool(x: int) -> int:
    """An async tool that doubles a number."""
    await asyncio.sleep(0.1)
    return x * 2

tool = dspy.Tool(async_tool)

# Option 1: Use context manager for temporary conversion
with dspy.context(allow_tool_async_sync_conversion=True):
    result = tool(x=5)  # Works in sync context
    print(result)  # 10

# Option 2: Configure globally
dspy.configure(allow_tool_async_sync_conversion=True)
result = tool(x=5)  # Now works everywhere
print(result)  # 10
```

For more details on async tools, see the [Tools documentation](../../learn/programming/tools.md#async-tools).

Note: When using `dspy.ReAct` with tools, calling `acall()` on the ReAct instance will automatically
execute all tools asynchronously using their `acall()` methods.

## Creating Custom Async DSPy Modules

To create your own async DSPy module, implement the `aforward()` method instead of `forward()`. This method
should contain your module's async logic. Here's an example of a custom module that chains two async operations:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

    async def aforward(self, question, **kwargs):
        # Execute predictions sequentially but asynchronously
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())
```
