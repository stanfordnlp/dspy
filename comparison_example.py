"""Example comparing Chain of Thought vs MetaLadder approaches."""
import os
from typing import Any, Dict, List, Optional

import dspy
from dspy import ChainOfThought, InputField, OutputField, Module, Predict
from dspy.signatures.signature import make_signature
from dspy.utils.dummies import DummyLM
from dspy.clients.lm import LM

from dspy.adapters.metaladder_adapter import MetaLadderAdapter

class MathSolver(dspy.Signature):
    """Signature for solving math word problems."""
    
    question = InputField(desc="A math word problem to solve")
    answer = OutputField(desc="The numerical answer with units")
    reasoning = OutputField(desc="Step by step reasoning process")

def solve_with_cot(lm: Any, question: str) -> Dict[str, str]:
    """Solve a problem using Chain of Thought reasoning.
    
    Args:
        lm: Language model to use
        question: Math problem to solve
        
    Returns:
        Dict containing answer and reasoning
    """
    # Create basic solver
    solver = ChainOfThought(MathSolver)
    dspy.settings.configure(lm=lm)
    
    # Get prediction
    pred = solver(question=question)
    return {
        "answer": pred.answer,
        "reasoning": pred.reasoning
    }

def solve_with_metaladder(lm: Any, question: str) -> Dict[str, Any]:
    """Solve a problem using MetaLadder approach.
    
    Args:
        lm: Language model to use
        question: Math problem to solve
        
    Returns:
        Dict containing answer and meta-problem details
    """
    # Create MetaLadder adapter
    adapter = MetaLadderAdapter(model=lm)
    dspy.settings.configure(lm=lm)
    
    # Get prediction and meta-problem
    pred = adapter(question=question)
    return {
        "answer": pred.answer,
        "meta_problem": adapter._meta_problems.get(question)
    }

def main() -> None:
    """Run comparison example."""
    # Initialize language model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    lm = LM(model="gpt-3.5-turbo", api_key=api_key)
    
    # Test problems of increasing complexity
    problems = [
        # Simple rate problem
        "If a car travels at 50 miles per hour for 3 hours, how far does it travel?",
        
        # Multi-step problem with unit conversion
        "A factory produces 120 widgets per hour and operates for 8 hours per day. If each widget requires 0.5 pounds of material, how many pounds of material are needed per week (5 days)?",
        
        # Problem requiring identifying relevant information
        "A store sells notebooks for $4 each and pens for $2 each. A student needs 3 notebooks and wants to spend exactly $20 in total. How many pens should they buy?",
        
        # Problem with distracting information
        "In a school library with 1000 books, 40% are fiction and 35% are non-fiction. If the remaining books are reference materials and 15 books are being repaired, how many reference books are available?"
    ]
    
    print("\n=== Comparing Problem-Solving Approaches ===\n")
    
    for i, problem in enumerate(problems, 1):
        print(f"Problem {i}:")
        print(f"Question: {problem}\n")
        
        try:
            # Solve with Chain of Thought
            print("Chain of Thought approach:")
            cot_result = solve_with_cot(lm, problem)
            print(f"Reasoning: {cot_result['reasoning']}")
            print(f"Answer: {cot_result['answer']}\n")
            
            # Solve with MetaLadder
            print("MetaLadder approach:")
            ml_result = solve_with_metaladder(lm, problem)
            meta = ml_result['meta_problem']
            print(f"Problem type: {meta.problem_type}")
            print(f"Meta-problem: {meta.meta_problem}")
            print(f"Restatement: {meta.restatement}")
            print(f"Answer: {ml_result['answer']}\n")
        except Exception as e:
            print(f"Error processing problem: {str(e)}\n")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main() 