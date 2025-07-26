#!/usr/bin/env python3
"""
Example: Using Claude Code with DSPy

This example demonstrates how to use Claude Code CLI for LLM inference with DSPy.
Claude Code provides direct access to Claude models through its CLI interface.

Prerequisites:
- Install Claude Code: npm install -g @anthropic-ai/claude-code
- Authenticate: claude auth login

The ClaudeCodeLM client uses the Claude CLI to execute queries and return responses
in OpenAI-compatible format for seamless DSPy integration.
"""

import dspy
from dspy.clients.claude_code_lm import create_claude_code_lm


def basic_claude_code_example():
    """Basic usage of Claude Code with DSPy."""
    print("=== Basic Claude Code Example ===")
    
    # Create Claude Code LM instance
    lm = create_claude_code_lm(
        claude_model="sonnet",  # Use Claude 3.5 Sonnet
        max_turns=1  # Single turn interaction
    )
    
    # Configure DSPy to use Claude Code
    dspy.configure(lm=lm)
    
    # Create a simple predictor
    predictor = dspy.Predict("question -> answer")
    
    # Test with a simple question
    result = predictor(question="What is the capital of France?")
    print(f"Question: What is the capital of France?")
    print(f"Answer: {result.answer}")
    print()


def multi_turn_claude_code_example():
    """Multi-turn conversation example with Claude Code."""
    print("=== Multi-turn Claude Code Example ===")
    
    # Create Claude Code LM with multiple turns enabled
    lm = create_claude_code_lm(
        claude_model="sonnet",
        max_turns=3  # Allow up to 3 agentic turns
    )
    
    dspy.configure(lm=lm)
    
    # Create a predictor for complex reasoning
    reasoning_predictor = dspy.Predict("problem -> reasoning, solution")
    
    # Test with a problem that might benefit from multi-turn reasoning
    problem = """
    A farmer has chickens and rabbits. In total, there are 20 heads and 56 legs.
    How many chickens and how many rabbits does the farmer have?
    """
    
    result = reasoning_predictor(problem=problem)
    print(f"Problem: {problem.strip()}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Solution: {result.solution}")
    print()


def chain_of_thought_example():
    """Chain of thought reasoning with Claude Code."""
    print("=== Chain of Thought Example ===")
    
    lm = create_claude_code_lm(claude_model="sonnet")
    dspy.configure(lm=lm)
    
    # Define a chain of thought signature
    class ChainOfThought(dspy.Signature):
        """Solve problems step by step with clear reasoning."""
        question = dspy.InputField()
        reasoning = dspy.OutputField(desc="step-by-step reasoning")
        answer = dspy.OutputField(desc="final answer")
    
    # Create predictor
    cot_predictor = dspy.Predict(ChainOfThought)
    
    # Test with a math problem
    question = "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"
    
    result = cot_predictor(question=question)
    print(f"Question: {question}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")
    print()


def direct_prompt_example():
    """Direct prompting without DSPy signatures."""
    print("=== Direct Prompt Example ===")
    
    lm = create_claude_code_lm(claude_model="sonnet")
    
    # Direct forward call
    prompt = "Explain the concept of machine learning in simple terms."
    response = lm.forward(prompt=prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage.total_tokens} tokens")
    print()


def messages_format_example():
    """Using messages format for conversation."""
    print("=== Messages Format Example ===")
    
    lm = create_claude_code_lm(claude_model="sonnet")
    
    # Messages format for conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant that explains complex topics simply."},
        {"role": "user", "content": "What is quantum computing?"},
        {"role": "assistant", "content": "Quantum computing uses quantum mechanics principles..."},
        {"role": "user", "content": "How is it different from classical computing?"}
    ]
    
    response = lm.forward(messages=messages)
    
    print("Conversation:")
    for msg in messages:
        print(f"{msg['role'].title()}: {msg['content']}")
    
    print(f"Claude Code Response: {response.choices[0].message.content}")
    print()


if __name__ == "__main__":
    try:
        print("Claude Code + DSPy Integration Examples\n")
        print("Note: Ensure Claude Code CLI is installed and authenticated")
        print("Install: npm install -g @anthropic-ai/claude-code")
        print("Auth: claude auth login\n")
        
        # Run examples
        basic_claude_code_example()
        multi_turn_claude_code_example()
        chain_of_thought_example()
        direct_prompt_example()
        messages_format_example()
        
        print("=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Install Claude Code: npm install -g @anthropic-ai/claude-code")
        print("2. Authenticate: claude auth login")
        print("3. Check DSPy installation: pip install dspy")