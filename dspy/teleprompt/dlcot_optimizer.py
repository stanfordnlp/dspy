import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.bootstrap import BootstrapFewShot

logger = logging.getLogger(__name__)


class DLCoT(Teleprompter):
    """Deconstructing Long Chain-of-Thought optimizer for enhanced distillation.

    This optimizer processes long CoT data through intelligent segmentation,
    redundancy elimination, and error state optimization to improve distillation efficiency.

    Based on research in "Deconstructing Long Chain-of-Thought: A Structured Reasoning
    Optimization Framework for Long CoT Distillation" (Luo et al., 2025).

    Key benefits:
    - Improved token efficiency by reducing redundancy in reasoning chains
    - Better performance across benchmarks by preserving essential verification steps
    - Enhanced reasoning coherence through structured optimization
    """

    def __init__(
        self,
        metric: Optional[Callable] = None,
        redundancy_threshold: float = 0.8,
        remove_incorrectness: bool = False,
        segment_config: Optional[Dict[str, Any]] = None,
        distillation_optimizer: Optional[Teleprompter] = None,
        num_threads: int = 6,
    ) -> None:
        """Initialize the DLCoT optimizer.

        Args:
            metric: Evaluation metric for filtering examples. Should return a boolean or numeric value.
            redundancy_threshold: Threshold for redundancy detection (0.0-1.0, higher means more sensitive)
            remove_incorrectness: Whether to remove incorrect reasoning paths
            segment_config: Configuration for the segmentation process (e.g., {"max_segments": 5})
            distillation_optimizer: Optimizer to use for distillation (defaults to BootstrapFewShot)
            num_threads: Number of threads for parallel processing
        """
        super().__init__()
        self.metric = metric
        self.redundancy_threshold = max(
            0.0, min(1.0, redundancy_threshold)
        )  # Ensure value is between 0 and 1
        self.remove_incorrectness = remove_incorrectness
        self.segment_config = segment_config or {"max_segments": 5}
        self.distillation_optimizer = distillation_optimizer
        self.num_threads = max(1, num_threads)  # Ensure at least 1 thread

    def compile(
        self,
        student: Program,
        trainset: List[Example],
        teacher: Optional[Program] = None,
    ) -> Program:
        """Compile the student module using DLCoT-enhanced distillation.

        This method implements the full DLCoT pipeline:
        1. Process dataset to optimize CoT reasoning chains
        2. Apply distillation using the processed data

        Args:
            student: Student module to optimize
            trainset: Training dataset with long CoT examples
            teacher: Optional teacher module (if None, student will be used as teacher)

        Returns:
            Optimized student module with improved reasoning capabilities
        """
        if not trainset:
            logger.warning(
                "[DLCoT] Empty training set provided. Returning original student module."
            )
            return student

        logger.info("[DLCoT] Processing long CoT data...")
        processed_trainset = self._process_dataset(trainset)

        logger.info("[DLCoT] Applying distillation with processed data...")
        # Use BootstrapFewShot by default, or the provided optimizer
        if self.distillation_optimizer is None:
            optimizer = BootstrapFewShot(metric=self.metric, max_bootstrapped_demos=4)
        else:
            optimizer = self.distillation_optimizer

        return optimizer.compile(
            student=student, trainset=processed_trainset, teacher=teacher
        )

    def _process_dataset(self, dataset: List[Example]) -> List[Example]:
        """Process the dataset according to the DLCoT methodology.

        Applies the DLCoT optimization to each example in the dataset.

        Args:
            dataset: Dataset containing long CoT examples

        Returns:
            Processed dataset with optimized CoT structure
        """
        processed_examples = []

        for example in dataset:
            processed_example = self._process_example(example)
            processed_examples.append(processed_example)

        return processed_examples

    def _process_example(self, example: Example) -> Example:
        """Process a single example according to the DLCoT methodology.

        Implements the five key steps from the paper:
        1. Macro-Structure Parsing: Split the solution into logical components
        2. Approach & Verification Parsing: Identify approaches and verification steps
        3. Redundancy Analysis: Identify and classify redundant and incorrect approaches
        4. Optimized Integration: Selectively remove redundant approaches
        5. Coherence Reconstruction: Ensure the processed content remains coherent

        Args:
            example: Original example with long CoT data

        Returns:
            Processed example with optimized CoT data
        """
        # Get the field containing the CoT content
        cot_field = self._find_cot_field(example)

        if not cot_field or not hasattr(example, cot_field):
            # No CoT data to process, return original example
            return example

        cot_content = getattr(example, cot_field)

        # Step 1: Segment the CoT into structural components
        segments = self._segment_cot(cot_content)

        # Step 2: Parse approaches and verification steps
        approaches, verifications = self._parse_approaches_and_verifications(segments)

        # Step 3: Analyze redundancy and correctness
        analyzed_approaches = self._analyze_redundancy_and_correctness(approaches)

        # Step 4: Optimize by removing redundancies
        optimized_segments = self._optimize_integration(segments, analyzed_approaches)

        # Step 5: Ensure coherence of the processed content
        final_content = self._ensure_coherence(optimized_segments)

        # Create a new example with the processed CoT content
        processed_example = example.copy()
        setattr(processed_example, cot_field, final_content)

        return processed_example

    def _find_cot_field(self, example: Example) -> Optional[str]:
        """Find the field in the example that contains the CoT content.

        This method looks for common field names that might contain Chain-of-Thought content.
        It first checks for a dedicated 'cot' field, then falls back to other candidates.

        Args:
            example: Example to analyze

        Returns:
            Name of the field containing CoT content, or None if not found
        """
        # If there's a 'cot' field, use that
        if hasattr(example, "cot") and isinstance(getattr(example, "cot"), str):
            return "cot"

        # This is a simplified heuristic - would need to be customized for specific datasets
        cot_field_candidates = [
            "answer",
            "response",
            "output",
            "completion",
            "reasoning",
        ]

        for field in cot_field_candidates:
            if hasattr(example, field) and isinstance(getattr(example, field), str):
                content = getattr(example, field)
                # Heuristic to identify CoT content (contains multiple steps or reasoning markers)
                if len(content.split("\n")) > 3 and (
                    "step" in content.lower() or "let me" in content.lower()
                ):
                    return field

        return None

    def _segment_cot(self, cot_content: str) -> Dict[str, str]:
        """Segment the CoT content into structural components.

        Splits the CoT content into logical segments according to the paper's methodology,
        including understanding the problem, exploring approaches, and verification.

        Args:
            cot_content: Raw CoT content as a string

        Returns:
            Dictionary with segmented components keyed by segment type
        """
        # In a real implementation, this would use more sophisticated segmentation,
        # potentially using an LLM to identify logical components as described in the paper

        segments = {
            "Question_Repeat": "",
            "Problem_Understand": "",
            "Approach_Explore": "",
            "Verify": "",
            "Conclusion": "",
        }

        # Simple heuristic segmentation
        lines = cot_content.split("\n")
        current_segment = "Question_Repeat"

        for line in lines:
            line_lower = line.lower()

            # Simple heuristic rules to determine segment transitions
            if "let's solve" in line_lower or "let me solve" in line_lower:
                current_segment = "Problem_Understand"
            elif any(
                marker in line_lower
                for marker in ["approach", "step 1", "first step", "calculate"]
            ):
                current_segment = "Approach_Explore"
            elif any(
                marker in line_lower
                for marker in ["verify", "check", "alternatively", "validation"]
            ):
                current_segment = "Verify"
            elif any(
                marker in line_lower
                for marker in [
                    "therefore",
                    "thus",
                    "final answer",
                    "conclusion",
                    "so",
                    "hence",
                ]
            ):
                current_segment = "Conclusion"

            segments[current_segment] += line + "\n"

        return segments

    def _parse_approaches_and_verifications(
        self, segments: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse approaches and verification steps from segmented content.

        Identifies distinct approach and verification blocks within the segments.

        Args:
            segments: Segmented CoT content

        Returns:
            Tuple of (approaches, verifications) where each item is a list of dictionaries
            containing metadata about the approaches and verifications
        """
        # This is a simplified implementation - would use more sophisticated parsing in practice
        approaches = []
        verifications = []

        approach_content = segments["Approach_Explore"]
        approach_blocks = self._split_into_approach_blocks(approach_content)

        for i, block in enumerate(approach_blocks):
            approaches.append(
                {
                    "id": f"approach_{i}",
                    "content": block,
                    "type": self._infer_approach_type(block),
                }
            )

        verification_content = segments["Verify"]
        verification_blocks = self._split_into_verification_blocks(verification_content)

        for i, block in enumerate(verification_blocks):
            verifications.append(
                {
                    "id": f"verification_{i}",
                    "content": block,
                    "type": self._infer_verification_type(block),
                }
            )

        return approaches, verifications

    def _split_into_approach_blocks(self, approach_content: str) -> List[str]:
        """Split approach content into distinct approach blocks.

        Approach blocks represent different methods or techniques used in the reasoning.

        Args:
            approach_content: Raw approach content as a string

        Returns:
            List of approach blocks, each as a string
        """
        # Simple heuristic: split by markers like "Alternative approach" or "Another way"
        markers = [
            "approach",
            "alternatively",
            "another way",
            "different",
            "method",
            "let's try",
        ]
        blocks = []
        current_block = []

        for line in approach_content.split("\n"):
            line_lower = line.lower()

            if (
                any(f"approach {i}" in line_lower for i in range(1, 10))
                and current_block
            ):
                blocks.append("\n".join(current_block))
                current_block = []
            elif any(marker in line_lower for marker in markers) and current_block:
                blocks.append("\n".join(current_block))
                current_block = []

            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        # If no blocks were found, treat the entire content as one block
        if not blocks and approach_content.strip():
            blocks = [approach_content]

        return blocks

    def _split_into_verification_blocks(self, verification_content: str) -> List[str]:
        """Split verification content into distinct verification blocks.

        Verification blocks represent different validation or checking steps.

        Args:
            verification_content: Raw verification content as a string

        Returns:
            List of verification blocks, each as a string
        """
        # Similar to approach blocks, but with verification-specific markers
        markers = ["verify", "check", "to confirm", "double-check", "validation"]
        blocks = []
        current_block = []

        for line in verification_content.split("\n"):
            line_lower = line.lower()

            if any(marker in line_lower for marker in markers) and current_block:
                blocks.append("\n".join(current_block))
                current_block = []

            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        # If no blocks were found, treat the entire content as one block
        if not blocks and verification_content.strip():
            blocks = [verification_content]

        return blocks

    def _infer_approach_type(self, approach_block: str) -> str:
        """Infer the type of approach from its content.

        Categorizes an approach based on its content and language.

        Args:
            approach_block: Content of an approach block as a string

        Returns:
            Type of approach as a string
        """
        # Simple heuristic classification
        approach_lower = approach_block.lower()

        if any(
            term in approach_lower
            for term in ["coordinate", "geometry", "angle", "degree"]
        ):
            return "coordinate_geometry"
        elif any(
            term in approach_lower
            for term in ["algebra", "equation", "variable", "solve for"]
        ):
            return "algebraic"
        elif any(
            term in approach_lower
            for term in [
                "arithmetic",
                "calculation",
                "multiply",
                "divide",
                "add",
                "subtract",
            ]
        ):
            return "arithmetic"
        elif any(
            term in approach_lower for term in ["factor", "factorize", "decompose"]
        ):
            return "factorization"
        elif any(
            term in approach_lower for term in ["formula", "substitution", "substitute"]
        ):
            return "formula_substitution"
        else:
            return "general"

    def _infer_verification_type(self, verification_block: str) -> str:
        """Infer the type of verification from its content.

        Categorizes a verification step based on its content and language.

        Args:
            verification_block: Content of a verification block as a string

        Returns:
            Type of verification as a string
        """
        # Simple heuristic classification
        verification_lower = verification_block.lower()

        if any(term in verification_lower for term in ["example", "instance", "case"]):
            return "example_check"
        elif any(
            term in verification_lower for term in ["substitute", "plug in", "insert"]
        ):
            return "substitution"
        elif any(
            term in verification_lower
            for term in ["double-check", "recalculate", "calculate again"]
        ):
            return "recalculation"
        elif any(term in verification_lower for term in ["unit", "dimension"]):
            return "unit_check"
        elif any(term in verification_lower for term in ["boundary", "edge", "limit"]):
            return "boundary_check"
        else:
            return "general_verification"

    def _analyze_redundancy_and_correctness(
        self, approaches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze approaches for redundancy and correctness.

        Identifies redundant approaches and attempts to assess correctness.

        Args:
            approaches: List of approach dictionaries

        Returns:
            Analyzed approaches with redundancy and correctness metadata
        """
        # In a real implementation, this would use more sophisticated analysis,
        # potentially using embedding similarity for redundancy detection

        analyzed_approaches = []

        # Track approach types and their counts
        approach_type_counts = {}

        for approach in approaches:
            approach_copy = approach.copy()
            approach_type = approach["type"]

            # Count occurrences of this approach type
            approach_type_counts[approach_type] = (
                approach_type_counts.get(approach_type, 0) + 1
            )

            # Determine if the approach is redundant (not the first of its type)
            is_redundant = approach_type_counts[approach_type] > 1
            redundancy_score = (approach_type_counts[approach_type] - 1) / max(
                1, len(approaches) - 1
            )

            approach_copy["is_redundant"] = is_redundant and (
                redundancy_score >= self.redundancy_threshold
            )
            approach_copy["redundancy_score"] = redundancy_score

            # In a real implementation, correctness would be inferred from analysis of the content
            # For this example, we use a simple heuristic
            has_error_markers = any(
                marker in approach["content"].lower()
                for marker in ["error", "mistake", "incorrect", "wrong"]
            )
            approach_copy["is_correct"] = not has_error_markers

            analyzed_approaches.append(approach_copy)

        return analyzed_approaches

    def _optimize_integration(
        self, segments: Dict[str, str], analyzed_approaches: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Optimize integration of approaches based on analysis.

        Selectively removes redundant or incorrect approaches based on configuration.

        Args:
            segments: Original segmented content
            analyzed_approaches: Analyzed approaches with metadata

        Returns:
            Optimized segmented content with redundancies removed
        """
        optimized_segments = segments.copy()

        # Filter out redundant approaches if needed
        filtered_approaches = []

        for approach in analyzed_approaches:
            should_keep = True

            # Remove redundant approaches based on threshold
            if approach["is_redundant"]:
                should_keep = False

            # Keep correct approaches even if redundant, if configured that way
            if (
                not should_keep
                and not self.remove_incorrectness
                and approach["is_correct"]
            ):
                should_keep = True

            # Always keep at least one approach, even if all are redundant
            if not filtered_approaches:
                should_keep = True

            if should_keep:
                filtered_approaches.append(approach)

        # Reconstruct the approach content
        optimized_approach_content = "\n\n".join(
            [approach["content"] for approach in filtered_approaches]
        )
        optimized_segments["Approach_Explore"] = optimized_approach_content

        return optimized_segments

    def _ensure_coherence(self, segments: Dict[str, str]) -> str:
        """Ensure coherence of the processed content.

        Reconstructs the content in a coherent way after optimization.

        Args:
            segments: Optimized segmented content

        Returns:
            Final coherent content as a string
        """
        # Build coherent content with appropriate transitions
        content_parts = []

        if segments["Question_Repeat"].strip():
            content_parts.append(segments["Question_Repeat"].strip())

        if segments["Problem_Understand"].strip():
            content_parts.append(segments["Problem_Understand"].strip())

        if segments["Approach_Explore"].strip():
            content_parts.append(segments["Approach_Explore"].strip())

        if segments["Verify"].strip():
            content_parts.append(segments["Verify"].strip())

        if segments["Conclusion"].strip():
            content_parts.append(segments["Conclusion"].strip())

        return "\n\n".join(content_parts)
