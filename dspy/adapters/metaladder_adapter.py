"""MetaLadder adapter implementation for DSPy."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetaProblem:
    """Represents a meta-level understanding of a problem.
    
    Attributes:
        problem_type: The type/category of the problem
        meta_problem: Abstract description of problem structure
        restatement: Problem restated using meta structure
    """
    problem_type: str
    meta_problem: str
    restatement: str


class MetaLadderAdapter:
    """Adapter that implements the MetaLadder approach for problem-solving."""
    
    def __init__(
        self,
        model: Any,
        use_shortcut: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cache_size: int = 1000,
        optimizer: Optional[Any] = None
    ) -> None:
        """Initialize the MetaLadder adapter.
        
        Args:
            model: The base model to use for predictions
            use_shortcut: Whether to skip meta-reasoning steps
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            cache_size: Size of meta-problem cache
            optimizer: Optional optimizer for the model
        """
        from dspy.predict import Predict
        from dspy.signatures.signature import make_signature
        
        logger.info("Initializing MetaLadder adapter")
        logger.info(f"Parameters: shortcut={use_shortcut}, temp={temperature}, max_tokens={max_tokens}")
        
        self.model = model
        self.use_shortcut = use_shortcut
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.optimizer = optimizer
        self._meta_problems: Dict[str, MetaProblem] = {}
        self._cache_size = cache_size
        
        # Create signatures for each step
        self.type_sig = make_signature(
            "question -> type",
            """Identify the mathematical operation type needed to solve this problem.
            Examples:
            - If the problem involves finding total from rate and time: 'multiplication'
            - If the problem involves sharing or distributing equally: 'division'
            - If the problem combines quantities: 'addition'
            - If the problem finds difference between quantities: 'subtraction'
            Output should be a single word in lowercase."""
        )
        self.meta_sig = make_signature(
            "type, question -> meta_problem",
            """Create a general template that captures the mathematical structure.
            Examples:
            For multiplication:
            - Rate × Time = Total (for rate problems)
            - Base × Multiplier = Product (for scaling problems)
            For division:
            - Total ÷ Number of parts = Size of each part
            - Whole ÷ Number of groups = Amount per group
            Keep it concise but clear."""
        )
        self.restate_sig = make_signature(
            "type, meta_problem, question -> restatement",
            """Rewrite the problem to match the meta-problem structure while preserving:
            1. All numerical values with their original units
            2. The specific context of the problem
            3. The exact mathematical relationship needed
            Example:
            Original: 'If a train travels at 60 mph for 2.5 hours, how far does it travel?'
            Restatement: 'Calculate the total distance when rate is 60 miles per hour and time is 2.5 hours.'"""
        )
        
        # Create predictors with specific temperatures
        self.type_predictor = Predict(self.type_sig, temperature=0.1)  # Low temp for consistent type identification
        self.meta_predictor = Predict(self.meta_sig, temperature=0.3)  # Moderate temp for meta-problem generation
        self.restate_predictor = Predict(self.restate_sig, temperature=0.1)  # Low temp for accurate restatement

    def __call__(self, **kwargs: Any) -> Any:
        """Call the adapter with the given inputs.
        
        Args:
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model output with the answer field
        """
        question = kwargs.get("question")
        if not question:
            raise ValueError("Question must be provided")
            
        answer, meta_problem = self.forward(question)
        logger.info(f"Final answer: {answer}")
        return type("Response", (), {"answer": answer})()

    def forward(self, question: str) -> Tuple[str, MetaProblem]:
        """Process a question using the MetaLadder approach.
        
        Args:
            question: The question to process
            
        Returns:
            Tuple of (answer, meta_problem)
        """
        logger.info(f"\nProcessing question: {question}")
        
        if self.use_shortcut:
            # Skip meta-reasoning and use model directly
            logger.info("Using shortcut path")
            response = self.model(messages=[{"role": "user", "content": question}])
            return response[0], MetaProblem("direct", "", "")
            
        # Check cache
        if question in self._meta_problems:
            logger.info("Using cached meta-problem")
            meta_problem = self._meta_problems[question]
        else:
            # Generate meta-problem components
            logger.info("Generating meta-problem components")
            
            # Step 1: Identify problem type
            problem_type = self.type_predictor(question=question).type
            logger.info(f"Identified problem type: {problem_type}")
            
            # Step 2: Generate meta-problem
            meta_problem = self.meta_predictor(
                type=problem_type,
                question=question
            ).meta_problem
            logger.info(f"Generated meta-problem: {meta_problem}")
            
            # Step 3: Restate problem
            restatement = self.restate_predictor(
                type=problem_type,
                meta_problem=meta_problem,
                question=question
            ).restatement
            logger.info(f"Generated restatement: {restatement}")
            
            meta_problem = MetaProblem(problem_type, meta_problem, restatement)
            
            # Update cache
            if len(self._meta_problems) >= self._cache_size:
                self._meta_problems.pop(next(iter(self._meta_problems)))
            self._meta_problems[question] = meta_problem
        
        # Get final answer using meta-problem
        logger.info("Getting final answer using meta-problem")
        response = self.model(messages=[{"role": "user", "content": meta_problem.restatement}])
        answer = response[0]
        logger.info(f"Final answer: {answer}")
        return answer, meta_problem

    def clear_cache(self) -> None:
        """Clear the meta-problem cache."""
        logger.info("Clearing meta-problem cache")
        self._meta_problems.clear() 