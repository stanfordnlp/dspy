"""Example usage of MetaLadderAdapter for mathematical reasoning."""

from typing import Any
from dspy.primitives.program import Module
from dspy.predict.predict import Predict
from dspy.signatures.signature import make_signature
from dspy.adapters.metaladder_adapter import MetaLadderAdapter
from dspy.clients.lm import LM

# Create a basic signature for our math solver
MathSolver = make_signature(
    "problem -> solution",
    "Given a mathematical problem, provide a step-by-step solution."
)

class SimpleMathModel(Module):
    """A simple model for solving math problems."""
    
    def __init__(self) -> None:
        """Initialize the model with a predictor."""
        super().__init__()
        self.predictor = Predict(MathSolver)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model."""
        return self.predictor(**kwargs)

def main() -> None:
    """Run an example using the MetaLadderAdapter."""
    # Initialize the language model
    lm = LM(model="gpt-3.5-turbo")
    
    # Create our math model
    model = SimpleMathModel()
    model.set_lm(lm)
    
    # Create the adapter
    adapter = MetaLadderAdapter(
        model=model,
        use_shortcut=False  # Use the full reasoning path
    )
    
    # Example math problem
    problem = "If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?"
    
    # Get the solution
    response, meta_problem = adapter.forward(problem)
    
    print("Problem Type:", meta_problem.problem_type)
    print("\nMeta Problem:", meta_problem.meta_problem)
    print("\nRestatement:", meta_problem.restatement)
    print("\nSolution:", response)

if __name__ == "__main__":
    main() 