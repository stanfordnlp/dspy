#!/usr/bin/env python
"""
Train a MetaLadder adapter for mathematical reasoning.

This script provides a standalone training process for the MetaLadder adapter,
allowing for faster debugging and iteration without running the full benchmark.

Usage:
    python train_metaladder.py --sample-size 10 --verbose
    python train_metaladder.py --model gpt-4o-mini --iterations 3
"""

import os
import sys
import time
import json
import random
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import dspy
from dspy.datasets.gsm8k import GSM8K
from dspy.clients.lm import LM
from dspy.adapters.metaladder_adapter import MetaLadderAdapter
from dspy.adapters.metaladder_trainer import MetaLadderTrainer, train_metaladder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def identify_problem_type(problem_text: str) -> str:
    """Identify the type of math problem based on keywords and patterns.
    
    Args:
        problem_text: The text of the math problem
        
    Returns:
        The identified problem type
    """
    problem_text = problem_text.lower()
    
    # Define keywords for different problem types with weights
    keywords = {
        "multiplication": {
            "times": 2, "multiply": 2, "product": 2, "twice": 1.5, "double": 1.5,
            "× ": 3, "*": 3, "multiplied by": 2
        },
        "division": {
            "divide": 2, "split": 1, "quotient": 2, "per": 1, "each": 0.5, "share": 1,
            "÷": 3, "/": 1.5, "divided by": 2, "out of": 1
        },
        "addition": {
            "add": 2, "sum": 2, "plus": 2, "total": 1.5, "combine": 1,
            "+": 3, "added to": 2, "increased by": 1.5
        },
        "subtraction": {
            "subtract": 2, "minus": 2, "difference": 2, "reduce": 1, "less": 1, "fewer": 1,
            "-": 3, "decreased by": 1.5, "subtracted from": 2
        },
        "percentage": {
            "percent": 2, "%": 3, "percentage": 2, "discount": 1.5, "interest": 1.5,
            "rate": 1, "tax": 1.5
        },
        "fractions": {
            "fraction": 2, "half": 1.5, "third": 1.5, "quarter": 1.5, "fifth": 1.5, "/": 1,
            "out of": 1, "portion": 1, "part": 0.5
        },
        "ratio": {
            "ratio": 3, "proportion": 2, "scale": 1.5, "to": 0.5, "for every": 1.5,
            "compared to": 1.5, "relative to": 1.5
        },
        "algebra": {
            "equation": 2, "solve for": 2, "variable": 2, "unknown": 1.5, "x": 1,
            "y": 1, "expression": 1.5, "formula": 1.5
        },
        "geometry": {
            "area": 2, "perimeter": 2, "volume": 2, "angle": 2, "circle": 1.5, "triangle": 1.5, "square": 1.5,
            "rectangle": 1.5, "diameter": 1.5, "radius": 1.5, "height": 1, "width": 1, "length": 1
        },
        "statistics": {
            "average": 2, "mean": 2, "median": 2, "mode": 2, "probability": 2,
            "chance": 1.5, "likelihood": 1.5, "data": 1, "sample": 1.5, "distribution": 1.5
        },
    }
    
    # Count weighted occurrences of keywords for each problem type
    type_scores = {}
    for problem_type, type_keywords in keywords.items():
        score = sum(weight for keyword, weight in type_keywords.items() if keyword in problem_text)
        type_scores[problem_type] = score
    
    # Check for number patterns that might indicate the problem type
    import re
    
    # Check for fractions (e.g., 1/2, 3/4)
    fraction_pattern = re.compile(r'\b\d+\s*/\s*\d+\b')
    if fraction_pattern.search(problem_text):
        type_scores["fractions"] += 2
    
    # Check for percentages (e.g., 50%, 75%)
    percentage_pattern = re.compile(r'\b\d+(\.\d+)?\s*%')
    if percentage_pattern.search(problem_text):
        type_scores["percentage"] += 2
    
    # Check for multiplication indicators (e.g., 5x, 3*4)
    mult_pattern = re.compile(r'\b\d+\s*[x×*]\s*\d+\b')
    if mult_pattern.search(problem_text):
        type_scores["multiplication"] += 2
    
    # Find the problem type with the highest score
    max_score = max(type_scores.values()) if type_scores else 0
    if max_score > 0:
        # Get all types with the highest score
        top_types = [t for t, s in type_scores.items() if s == max_score]
        if len(top_types) == 1:
            return top_types[0]
        else:
            # If there's a tie, prefer certain problem types in this order
            priority_order = ["fractions", "percentage", "division", "multiplication", 
                             "addition", "subtraction", "algebra", "geometry", 
                             "statistics", "ratio"]
            for p_type in priority_order:
                if p_type in top_types:
                    return p_type
    
    # Default to "other" if no keywords are found or no clear winner
    return "other"


def load_gsm8k_problems(sample_size: int = 10, balanced: bool = False) -> Tuple[List[str], List[str]]:
    """Load problems from the GSM8K dataset.
    
    Args:
        sample_size: Number of problems to load
        balanced: Whether to ensure a balanced distribution of problem types
        
    Returns:
        Tuple of (questions, answers)
    """
    logger.info(f"Loading {sample_size} problems from GSM8K dataset...")
    gsm8k = GSM8K()
    test_data = gsm8k.test
    
    # Convert to list of questions and answers
    problems = []
    answers = []
    
    if balanced:
        # Identify problem types using a simple heuristic
        problem_types: Dict[str, List[Any]] = {}
        for example in test_data:
            # Use our problem type identification logic
            problem_type = identify_problem_type(example.question)
            if problem_type not in problem_types:
                problem_types[problem_type] = []
            problem_types[problem_type].append(example)
        
        logger.info(f"Found {len(problem_types)} problem types: {', '.join(problem_types.keys())}")
        
        # Calculate how many problems to take from each type
        problems_per_type = max(1, sample_size // len(problem_types))
        logger.info(f"Selecting approximately {problems_per_type} problems per type")
        
        # Sample from each problem type
        import random
        random.seed(42)  # For reproducibility
        balanced_sample = []
        for problem_type, examples in problem_types.items():
            # Take min of available examples or problems_per_type
            type_sample_size = min(len(examples), problems_per_type)
            type_sample = random.sample(examples, type_sample_size)
            balanced_sample.extend(type_sample)
            logger.info(f"Selected {len(type_sample)} '{problem_type}' problems")
        
        # If we need more problems to reach sample_size, add random problems
        if len(balanced_sample) < sample_size:
            remaining = sample_size - len(balanced_sample)
            # Get examples not already in balanced_sample
            remaining_examples = [ex for ex in test_data if ex not in balanced_sample]
            if remaining_examples:
                additional = random.sample(remaining_examples, min(remaining, len(remaining_examples)))
                balanced_sample.extend(additional)
                logger.info(f"Added {len(additional)} additional random problems to reach target size")
        
        # If we have too many problems, trim to sample_size
        if len(balanced_sample) > sample_size:
            balanced_sample = balanced_sample[:sample_size]
        
        # Extract questions and answers from the balanced sample
        problems = [example.question for example in balanced_sample]
        answers = [example.answer for example in balanced_sample]
    else:
        # Take a random sample for training
        if sample_size > 0 and sample_size < len(test_data):
            import random
            random.seed(42)  # For reproducibility
            sampled_data = random.sample(list(test_data), sample_size)
        else:
            sampled_data = test_data
        
        # Extract questions and answers
        problems = [example.question for example in sampled_data]
        answers = [example.answer for example in sampled_data]
    
    logger.info(f"Loaded {len(problems)} problems with answers")
    
    return problems, answers

def create_chain_of_thought_solver() -> dspy.ChainOfThought:
    """
    Create a Chain of Thought solver.
    
    Returns:
        ChainOfThought solver
    """
    # Define the signature for the math solver
    class MathSolver(dspy.Signature):
        """Solve a math problem step by step."""
        question = dspy.InputField(desc="The math problem to solve")
        answer = dspy.OutputField(desc="The final answer to the math problem")
    
    # Create a Chain of Thought solver
    cot_solver = dspy.ChainOfThought(MathSolver)
    
    return cot_solver

def main() -> None:
    """Main function to train the MetaLadder adapter.
    
    This function handles the entire training pipeline:
    1. Parses command line arguments
    2. Sets up the language model
    3. Loads training data from GSM8K
    4. Creates and trains the MetaLadder adapter
    5. Tests the trained adapter on sample problems
    6. Optionally saves the trained adapter
    """
    
    # Model options
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                        help="Model to use (e.g., gpt-4o-mini, gpt-3.5-turbo, gpt-4)")
    parser.add_argument("--api-base", type=str, default=None, 
                        help="Base URL for API requests (optional)")
    
    # Training options
    parser.add_argument("--iterations", type=int, default=3, help="Number of training iterations")
    parser.add_argument("--bootstrap-examples", type=int, default=5, help="Number of bootstrap examples")
    parser.add_argument("--variation-temp", type=float, default=0.7, help="Temperature for variation generation")
    
    # Hybrid adapter options
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid adapter combining MetaLadder and Chain of Thought")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, 
                        help="Confidence threshold for hybrid adapter (lower values favor MetaLadder)")
    parser.add_argument("--cache-building-ratio", type=float, default=0.3,
                        help="Ratio of problems to always solve with MetaLadder to build cache (0.0-1.0)")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--save", action="store_true", help="Save the trained adapter to disk")
    parser.add_argument("--output-dir", type=str, default="trained_models", help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Set up the language model with API key
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("Please set the OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Initialize the language model
    model_name = args.model
    logger.info(f"Initializing language model: {model_name}")
    
    # Set up the language model with optional API base URL
    lm_kwargs = {
        "model": model_name,
        "api_key": os.environ.get("OPENAI_API_KEY")
    }
    
    if args.api_base:
        logger.info(f"Using custom API base URL: {args.api_base}")
        lm_kwargs["api_base"] = args.api_base
    
    lm = dspy.OpenAI(**lm_kwargs)
    
    # Load GSM8K dataset with balanced problem types if requested
    try:
        problems, answers = load_gsm8k_problems(
            sample_size=args.sample_size,
            balanced=args.balanced
        )
        
        if args.verbose:
            # Display sample problems
            logger.info("Sample problems:")
            for i in range(min(3, len(problems))):
                logger.info(f"Problem {i+1}: {problems[i][:100]}...")
                logger.info(f"Answer: {answers[i]}\n")
                
            # Analyze problem type distribution
            problem_types = {}
            for problem in problems:
                problem_type = identify_problem_type(problem)
                problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
            
            logger.info("Problem type distribution:")
            for problem_type, count in problem_types.items():
                percentage = (count / len(problems)) * 100
                logger.info(f"- {problem_type}: {count} problems ({percentage:.1f}%)")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    # Configure the language model with temperature
    if args.temperature != 0.7:  # Only log if not using default
        logger.info(f"Setting model temperature to {args.temperature}")
        lm.temperature = args.temperature
    
    # Create a Chain of Thought solver with proper typing
    class MathSolver(dspy.Signature):
        """Signature for solving math problems."""
        question: str = dspy.InputField(desc="The math problem to solve")
        answer: str = dspy.OutputField(desc="The numerical answer with units")
        reasoning: str = dspy.OutputField(desc="Step by step reasoning process")
    
    cot_solver = dspy.ChainOfThought(MathSolver)
    
    # Create a MetaLadder trainer with configurable parameters
    logger.info(f"Creating trainer with {args.iterations} iterations, "
               f"{args.bootstrap_examples} bootstrap examples, "
               f"temperature {args.variation_temp}")
    
    trainer = MetaLadderTrainer(
        model=cot_solver,
        num_iterations=args.iterations,
        num_bootstrap_examples=args.bootstrap_examples,
        temperature=args.variation_temp
    )
    
    # Train the adapter using the train_metaladder function with custom parameters
    logger.info("Starting training...")
    start_time = time.time()
    try:
        # Create a MetaLadder adapter with the specified parameters
        adapter = MetaLadderAdapter(
            model=cot_solver,
            use_analogical_reasoning=True,
            use_shortcut=False,
            temperature=args.temperature
        )
        
        # Train the adapter
        trained_adapter = trainer.train(adapter, problems, answers)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        
        # Log information about the trained adapter
        logger.info(f"Trained adapter information:")
        logger.info(f"- Cache size: {len(trained_adapter._meta_problems)} problems")
        logger.info(f"- Analogical reasoning: {trained_adapter.use_analogical_reasoning}")
        logger.info(f"- Shortcut enabled: {trained_adapter.use_shortcut}")
        
        # Implement hybrid approach that combines Chain of Thought and MetaLadder
        logger.info("Creating hybrid adapter that combines Chain of Thought and MetaLadder...")
        
        class HybridAdapter:
            """A hybrid adapter that combines Chain of Thought and MetaLadder approaches.
            
            This adapter uses MetaLadder's meta-problem generation and analogical reasoning
            but falls back to Chain of Thought for direct solving when appropriate.
            
            Args:
                metaladder: The trained MetaLadder adapter
                cot: The Chain of Thought solver
                confidence_threshold: Threshold for using MetaLadder vs. CoT
            """
            
            def __init__(self, metaladder: MetaLadderAdapter, cot: dspy.ChainOfThought, 
                         confidence_threshold: float = 0.5, cache_building_ratio: float = 0.3) -> None:
                self.metaladder = metaladder
                self.cot = cot
                self.confidence_threshold = confidence_threshold
                self.cache_building_ratio = cache_building_ratio
                self.stats = {
                    "metaladder_used": 0, 
                    "cot_used": 0,
                    "cache_building": 0,
                    "confidence_based": 0,
                    "confidence_scores": []
                }
                logger.info(f"Hybrid adapter initialized with confidence threshold: {confidence_threshold}")
                logger.info(f"Cache building ratio: {cache_building_ratio}")
                logger.info(f"Lower threshold values favor using MetaLadder more frequently")
            
            def calculate_similarity(self, problem1: str, problem2: str) -> float:
                """Calculate a similarity score between two problems using multiple metrics.
                
                Args:
                    problem1: First problem text
                    problem2: Second problem text
                    
                Returns:
                    Similarity score between 0.0 and 1.0
                """
                # Normalize and clean the problems
                p1 = problem1.lower()
                p2 = problem2.lower()
                
                # Calculate word overlap (Jaccard similarity)
                words1 = set(p1.split())
                words2 = set(p2.split())
                
                if not words1 or not words2:
                    return 0.0
                    
                # Calculate Jaccard similarity for words
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                jaccard = intersection / union if union > 0 else 0.0
                
                # Check for number similarity
                import re
                numbers1 = set(re.findall(r'\d+\.?\d*', p1))
                numbers2 = set(re.findall(r'\d+\.?\d*', p2))
                
                # Calculate number similarity
                num_intersection = len(numbers1.intersection(numbers2))
                num_union = len(numbers1.union(numbers2))
                number_sim = num_intersection / num_union if num_union > 0 else 0.0
                
                # Check for key phrases that indicate similar problem structures
                key_phrases = [
                    "how many", "what is", "calculate", "find", "solve", 
                    "total", "average", "percent", "ratio", "fraction",
                    "times", "divided by", "plus", "minus", "multiply", "add", "subtract"
                ]
                
                phrase_count1 = sum(1 for phrase in key_phrases if phrase in p1)
                phrase_count2 = sum(1 for phrase in key_phrases if phrase in p2)
                phrase_sim = 0.0
                if phrase_count1 > 0 and phrase_count2 > 0:
                    phrase_sim = min(phrase_count1, phrase_count2) / max(phrase_count1, phrase_count2)
                
                # Combine the similarity scores with weights
                combined_sim = (jaccard * 0.5) + (number_sim * 0.3) + (phrase_sim * 0.2)
                
                return combined_sim
            
            def forward(self, question: str) -> Tuple[str, Optional[Any]]:
                """Process a question using the hybrid approach.
                
                Args:
                    question: The input question to solve
                    
                Returns:
                    Tuple of (answer, meta_problem)
                """
                # First, check if we have a similar problem in the MetaLadder cache
                best_meta_problem = None
                best_confidence = 0.0  # Initialize confidence score
                problem_type = identify_problem_type(question)
                
                # Always try MetaLadder first for new problems to build the cache
                # or periodically to ensure we're building a diverse cache
                if len(self.metaladder._meta_problems) < 5 or random.random() < self.cache_building_ratio:
                    logger.info("Using MetaLadder approach (building cache)")
                    answer, meta_problem = self.metaladder.forward(question)
                    self.stats["metaladder_used"] += 1
                    self.stats["cache_building"] += 1
                    return answer, meta_problem
                
                # Look for similar problems in the cache
                if self.metaladder._meta_problems:
                    for problem_key, cached_meta_problem in self.metaladder._meta_problems.items():
                        try:
                            # Get the original problem from the meta-problem
                            if hasattr(cached_meta_problem, 'original_problem'):
                                original_problem = cached_meta_problem.original_problem
                            else:
                                continue
                                
                            # Calculate similarity based on problem type and content
                            cached_type = identify_problem_type(original_problem)
                            type_match = problem_type == cached_type
                            
                            # Calculate text similarity
                            text_similarity = self.calculate_similarity(question, original_problem)
                            
                            # Combined confidence score
                            confidence = text_similarity
                            if type_match:
                                confidence += 0.4  # Boost confidence if problem types match
                                
                            # Additional boost for very similar problems
                            if text_similarity > 0.5:
                                confidence += 0.2  # Extra boost for high text similarity
                            
                            # Keep track of the best match
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_meta_problem = cached_meta_problem
                                
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"Error accessing meta-problem: {e}")
                            continue
                
                # If we found a similar problem with high confidence, use MetaLadder
                # Initialize meta_problem to None to avoid UnboundLocalError
                meta_problem = None
                
                # Record the confidence score for analysis
                self.stats["confidence_scores"].append(best_confidence)
                
                if best_meta_problem and best_confidence >= self.confidence_threshold:
                    logger.info(f"Using MetaLadder approach (confidence: {best_confidence:.2f})")
                    answer, meta_problem = self.metaladder.forward(question)
                    self.stats["metaladder_used"] += 1
                    self.stats["confidence_based"] += 1
                    
                    # Log the meta-problem that matched
                    if hasattr(best_meta_problem, 'original_problem'):
                        logger.debug(f"Matched with problem: {best_meta_problem.original_problem[:100]}...")
                else:
                    # Otherwise, use Chain of Thought for direct solving
                    if best_confidence > 0:
                        logger.info(f"Using Chain of Thought approach (confidence {best_confidence:.2f} below threshold {self.confidence_threshold})")
                    else:
                        logger.info("Using Chain of Thought approach (no similar problem found)")
                    prediction = self.cot(question=question)
                    answer = prediction.answer
                    self.stats["cot_used"] += 1
                
                return answer, meta_problem
        
        # Create the hybrid adapter if requested
        if args.hybrid:
            hybrid_adapter = HybridAdapter(
                trained_adapter, 
                cot_solver, 
                args.confidence_threshold,
                args.cache_building_ratio
            )
            logger.info("Hybrid adapter created successfully")
            adapter_to_use = hybrid_adapter
        else:
            logger.info("Using pure MetaLadder adapter (no hybrid)")
            adapter_to_use = trained_adapter
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test the trained adapter on separate test problems
    logger.info("Testing trained adapter on test problems:")
    
    # Get test problems (either from the training set or new ones)
    if args.test_size > 0:
        if args.test_size <= len(problems) and args.test_size < 10:
            # Use a subset of training problems for testing
            test_problems = problems[-args.test_size:]
            test_answers = answers[-args.test_size:]
            logger.info(f"Using {args.test_size} problems from the training set for testing")
        else:
            # Load new problems for testing
            test_problems, test_answers = load_gsm8k_problems(
                sample_size=args.test_size,
                balanced=args.balanced
            )
            logger.info(f"Loaded {len(test_problems)} new problems for testing")
    else:
        # Default to a few problems from the training set
        test_size = min(5, len(problems))
        test_problems = problems[-test_size:]
        test_answers = answers[-test_size:]
        logger.info(f"Using {test_size} problems from the training set for testing")
    
    # Run tests and collect metrics
    correct = 0
    total = len(test_problems)
    latencies = []
    problem_type_results = {}
    
    for i in range(total):
        problem = test_problems[i]
        expected = test_answers[i]
        problem_type = identify_problem_type(problem)
        
        logger.info(f"\nTesting problem {i+1} (type: {problem_type}): {problem[:100]}...")
        start_time = time.time()
        # Use the selected adapter for inference
        answer, meta_problem = adapter_to_use.forward(problem)
        inference_time = time.time() - start_time
        latencies.append(inference_time)
        
        logger.info(f"Answer: {answer}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Time: {inference_time:.2f}s")
        
        # Track results by problem type
        if problem_type not in problem_type_results:
            problem_type_results[problem_type] = {"correct": 0, "total": 0}
        problem_type_results[problem_type]["total"] += 1
        
        # Normalize answers for comparison (extract numbers and units)
        def normalize_answer(ans):
            # Convert to lowercase
            ans = ans.lower().strip()
            # Extract numbers using regex
            import re
            numbers = re.findall(r'\d+', ans)
            if numbers:
                return numbers[0]  # Return the first number found
            return ans
            
        normalized_answer = normalize_answer(answer)
        normalized_expected = normalize_answer(expected)
        
        # Compare normalized answers
        is_correct = normalized_answer == normalized_expected
        if is_correct:
            correct += 1
            logger.info("Result: Correct")
        else:
            logger.info(f"Result: Incorrect (normalized: {normalized_answer} vs {normalized_expected})")
    
    # Report accuracy on test problems
    accuracy = (correct / total) * 100 if total > 0 else 0
    logger.info(f"\nAccuracy on {total} test problems: {accuracy:.2f}%")
    
    # Report hybrid adapter usage statistics
    if args.hybrid and 'hybrid_adapter' in locals():
        total_problems = hybrid_adapter.stats["metaladder_used"] + hybrid_adapter.stats["cot_used"]
        metaladder_percent = (hybrid_adapter.stats["metaladder_used"] / total_problems * 100) if total_problems > 0 else 0
        cot_percent = (hybrid_adapter.stats["cot_used"] / total_problems * 100) if total_problems > 0 else 0
        cache_building_percent = (hybrid_adapter.stats["cache_building"] / hybrid_adapter.stats["metaladder_used"] * 100) if hybrid_adapter.stats["metaladder_used"] > 0 else 0
        confidence_based_percent = (hybrid_adapter.stats["confidence_based"] / hybrid_adapter.stats["metaladder_used"] * 100) if hybrid_adapter.stats["metaladder_used"] > 0 else 0
        
        # Calculate average confidence score if available
        confidence_scores = hybrid_adapter.stats["confidence_scores"]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        logger.info(f"\nHybrid adapter usage statistics:")
        logger.info(f"- MetaLadder approach: {hybrid_adapter.stats['metaladder_used']} problems ({metaladder_percent:.1f}%)")
        logger.info(f"  - Cache building: {hybrid_adapter.stats['cache_building']} problems ({cache_building_percent:.1f}% of MetaLadder usage)")
        logger.info(f"  - Confidence-based: {hybrid_adapter.stats['confidence_based']} problems ({confidence_based_percent:.1f}% of MetaLadder usage)")
        logger.info(f"- Chain of Thought approach: {hybrid_adapter.stats['cot_used']} problems ({cot_percent:.1f}%)")
        logger.info(f"- Confidence threshold: {args.confidence_threshold}")
        logger.info(f"- Average confidence score: {avg_confidence:.2f}")
        logger.info(f"- Cache building ratio: {args.cache_building_ratio}")
    
    # Calculate and report detailed metrics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        median_latency = sorted(latencies)[len(latencies) // 2]
        logger.info(f"\nLatency metrics:")
        logger.info(f"- Average: {avg_latency:.2f}s")
        logger.info(f"- Median: {median_latency:.2f}s")
        logger.info(f"- Min: {min(latencies):.2f}s")
        logger.info(f"- Max: {max(latencies):.2f}s")
    
    # Report problem type accuracy
    if problem_type_results:
        logger.info(f"\nAccuracy by problem type:")
        for problem_type, results in problem_type_results.items():
            if results["total"] > 0:
                type_accuracy = (results["correct"] / results["total"]) * 100
                logger.info(f"- {problem_type}: {type_accuracy:.2f}% ({results['correct']}/{results['total']})")
    
    # Save the trained adapter and metrics if requested
    if args.save:
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            # Save metadata about the training
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            metadata = {
                "model": args.model,
                "iterations": args.iterations,
                "bootstrap_examples": args.bootstrap_examples,
                "temperature": args.temperature,
                "variation_temperature": args.variation_temp,
                "use_analogical": args.use_analogical,
                "use_shortcut": args.use_shortcut,
                "balanced_training": args.balanced,
                "training_problems": len(problems),
                "test_problems": len(test_problems),
                "timestamp": timestamp,
                "accuracy": accuracy,
                "latency": {
                    "avg": avg_latency if latencies else None,
                    "median": median_latency if latencies else None,
                    "min": min(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None
                },
                "problem_type_results": {
                    ptype: {
                        "accuracy": (results["correct"] / results["total"]) * 100 if results["total"] > 0 else 0,
                        "correct": results["correct"],
                        "total": results["total"]
                    } for ptype, results in problem_type_results.items()
                },
                "hybrid_stats": hybrid_adapter.stats if args.hybrid and 'hybrid_adapter' in locals() else None,
                "hybrid_mode": args.hybrid,
                "confidence_threshold": args.confidence_threshold if args.hybrid else None,
                "cache_building_ratio": args.cache_building_ratio if args.hybrid else None,
                "model": args.model
            }
            
            metadata_path = os.path.join(args.output_dir, f"metaladder_training_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\nSaved training metadata to {metadata_path}")
            logger.info("Note: Model weights are not saved, only the training configuration")
        except Exception as e:
            logger.error(f"Error saving trained adapter: {str(e)}")
    
    logger.info("\nTraining process completed successfully")

if __name__ == "__main__":
    main()
