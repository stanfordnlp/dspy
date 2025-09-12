import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
from dspy import BootstrapFewShot, COPRO, Evaluate, MIPROv2


# Data models for structured output
@dataclass
class VocabTranslationResult:
    """Result of vocabulary translation evaluation."""

    is_correct: int
    correct_translation: str
    explanation: str
    target_word: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "correct_translation": self.correct_translation,
            "explanation": self.explanation,
            "target_word": self.target_word,
        }


# Enhanced signature with more detailed instructions
class DetailedVocabularyTranslation(dspy.Signature):
    """
    Evaluate vocabulary translation as an expert French tutor.
    Consider common synonyms and variations.
    Don't mark wrong if user didn't provide every possible translation.
    If the word seems non-standard or rare, mark as correct.
    """

    source_word: str = dspy.InputField(desc="Word to translate from source language")
    target_word: str = dspy.InputField(desc="User's translation attempt")
    source_lang: str = dspy.InputField(desc="Source language")
    target_lang: str = dspy.InputField(desc="Target language")

    is_correct: int = dspy.OutputField(desc="1 for correct, 0 for incorrect")
    correct_translation: str = dspy.OutputField(desc="Best translation of the source word")
    explanation: str = dspy.OutputField(desc="Why the translation is correct/incorrect")


class VocabularyTranslationAgent(dspy.Module):
    """DSPy module for evaluating vocabulary translations with structured output."""

    def __init__(self):
        super().__init__()

        # Use TypedPredictor for structured output
        self.evaluate = DetailedVocabularyTranslation

    def forward(self, word: str, translation: str, from_french_to_english: bool = True) -> VocabTranslationResult:
        """
        Evaluate if a vocabulary translation is correct.

        Args:
            word: The word to translate
            translation: The user's translation
            from_french_to_english: Direction of translation

        Returns:
            VocabTranslationResult with evaluation details
        """
        # Determine source and target languages
        if from_french_to_english:
            source_lang = "French"
            target_lang = "English"
            source_word = word
            target_word = translation
        else:
            source_lang = "English"
            target_lang = "French"
            source_word = translation
            target_word = word

        # Get structured response directly
        result = self.evaluate(
            source_word=source_word, target_word=target_word, source_lang=source_lang, target_lang=target_lang
        )

        return VocabTranslationResult(
            is_correct=result.is_correct,
            correct_translation=result.correct_translation,
            explanation=result.explanation,
            target_word=target_word,
        )


class EnhancedVocabularyAgent(dspy.Module):
    """Enhanced vocabulary translation agent with chain of thought reasoning."""

    def __init__(self, model: Optional[str] = None):
        super().__init__()

        # Use ChainOfThought for better reasoning
        self.evaluate = dspy.ChainOfThought(DetailedVocabularyTranslation)

        if model:
            self.lm = dspy.LM(model=model)
            dspy.configure(lm=self.lm)

    def forward(self, word: str, translation: str, from_french_to_english: bool = True) -> VocabTranslationResult:
        """Evaluate with enhanced reasoning."""
        # Prepare languages and words
        if from_french_to_english:
            source_lang, target_lang = "French", "English"
            source_word, target_word = word, translation
        else:
            source_lang, target_lang = "English", "French"
            source_word, target_word = translation, word

        # Evaluate with reasoning
        result = self.evaluate(
            source_word=source_word, target_word=target_word, source_lang=source_lang, target_lang=target_lang
        )

        return VocabTranslationResult(
            is_correct=result.is_correct,
            correct_translation=result.correct_translation,
            explanation=result.explanation,
            target_word=target_word,
        )


def load_examples_from_csv(csv_path: str = "vocab_examples.csv") -> List[dspy.Example]:
    """Load vocabulary examples from CSV file."""
    examples = []

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = dspy.Example(
                source_word=row['source_word'],
                target_word=row['target_word'],
                source_lang=row['source_lang'],
                target_lang=row['target_lang'],
                expected=int(row['expected'])
            ).with_inputs("source_word", "target_word", "source_lang", "target_lang")
            examples.append(example)

    return examples


def score_translation(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """Score a translation result against expected outcome."""
    # Handle different calling signatures from GEPA vs regular evaluation
    if hasattr(prediction, 'is_correct'):
        return example["expected"] == prediction.is_correct
    else:
        # For GEPA, prediction might be the raw output
        return 0.0  # Default fallback


def split_dataset(examples: List[dspy.Example], train_ratio: float = 0.8) -> tuple[
    List[dspy.Example], List[dspy.Example]]:
    """Split dataset into training and validation sets."""
    import random
    random.seed(42)  # For reproducible splits

    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]

    return train_set, val_set


def run_optimization(examples: List[dspy.Example], optimizer_type: str = "bootstrap") -> dspy.Module:
    """Run DSPy optimization with different optimizers."""

    # Split dataset
    train_set, val_set = split_dataset(examples)
    print(f"Training set: {len(train_set)} examples")
    print(f"Validation set: {len(val_set)} examples")

    # Create student program (the one to optimize)
    student = dspy.ChainOfThought(DetailedVocabularyTranslation)

    # Setup optimizer based on type
    if optimizer_type == "bootstrap":
        print("Using BootstrapFewShot optimizer...")
        optimizer = BootstrapFewShot(
            metric=score_translation,
            max_bootstrapped_demos=8,  # Number of examples to bootstrap
            max_labeled_demos=4,  # Max labeled demonstrations
        )
    elif optimizer_type == "copro":
        print("Using COPRO optimizer...")
        optimizer = COPRO(
            metric=score_translation,
            breadth=10,
            depth=3,
            init_temperature=1.0
        )
    elif optimizer_type == "miprov2":
        print("Using MIPROv2 optimizer...")
        optimizer = MIPROv2(
            metric=score_translation,
            auto="light"  # Use light auto-configuration
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print(f"Starting {optimizer_type} optimization...")

    # Run optimization
    if optimizer_type == "bootstrap":
        optimized_program = optimizer.compile(
            student=student,
            trainset=train_set  # BootstrapFewShot doesn't use valset
        )
    elif optimizer_type == "copro":
        optimized_program = optimizer.compile(
            student=student,
            trainset=train_set,  # COPRO doesn't use valset
            eval_kwargs={}  # Required parameter for COPRO
        )
    else:
        optimized_program = optimizer.compile(
            student=student,
            trainset=train_set,  # Use subset for faster training
            valset=val_set  # Use subset for validation
        )

    print(f"{optimizer_type} optimization completed!")

    # Evaluate on full validation set
    evaluator = Evaluate(devset=val_set, num_threads=1, display_progress=True)

    print("\nEvaluating original program:")
    original_result = evaluator(student, metric=score_translation)
    original_score = original_result if isinstance(original_result, (int, float)) else original_result.score

    print("\nEvaluating optimized program:")
    optimized_result = evaluator(optimized_program, metric=score_translation)
    optimized_score = optimized_result if isinstance(optimized_result, (int, float)) else optimized_result.score

    print(f"\nOptimization Results ({optimizer_type}):")
    print(f"Original accuracy: {original_score:.1%}")
    print(f"Optimized accuracy: {optimized_score:.1%}")
    print(f"Improvement: {(optimized_score - original_score) * 100:.1f} percentage points")

    return optimized_program


# Example usage and testing
if __name__ == "__main__":
    # Setup Gemma model through LM Studio using standard DSPy LM
    import litellm

    litellm.set_verbose = False

    lm = dspy.LM(
        model="openai/gemma-3-1b-it",
        api_base="http://localhost:1234/v1",
        api_key="dummy",
        temperature=1.0,
        top_p=0.95,
        min_p=0.0,
        frequency_penalty=1,
        # max_tokens=512
    )
    dspy.configure(lm=lm)
    # Load examples from CSV
    examples = load_examples_from_csv()
    print(f"Loaded {len(examples)} examples from CSV")

    # Choose mode: basic evaluation or GEPA optimization
    import sys

    if len(sys.argv) > 1 and sys.argv[1].startswith("--optimize"):
        # Parse optimizer type
        optimizer_type = "bootstrap"  # default
        if len(sys.argv) > 2:
            optimizer_type = sys.argv[2]
        elif "=" in sys.argv[1]:
            optimizer_type = sys.argv[1].split("=")[1]

        # Run optimization
        optimized_program = run_optimization(examples, optimizer_type)

        # Save optimized program (optional)
        # optimized_program.save(f'optimized_vocab_agent_{optimizer_type}.json')

    else:
        # Basic evaluation
        agent = dspy.Predict(DetailedVocabularyTranslation)
        evaluator = Evaluate(devset=examples, num_threads=1, display_progress=True, display_table=True)
        score = evaluator(agent, metric=score_translation).score
        print(f"\nBaseline accuracy: {score:.1%}")
        print("\nOptimization options:")
        print("  python vocab_agent.py --optimize bootstrap  # BootstrapFewShot (recommended)")
        print("  python vocab_agent.py --optimize copro      # COPRO")
        print("  python vocab_agent.py --optimize miprov2    # MIPROv2")
