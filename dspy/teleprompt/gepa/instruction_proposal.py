from typing import Any

from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.types.base_type import Type
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample


class GenerateEnhancedMultimodalInstructionFromFeedback(dspy.Signature):
    """I provided an assistant with instructions to perform a task involving visual content, but the assistant's performance needs improvement based on the examples and feedback below.

    Your task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback, with particular attention to how visual and textual information should be analyzed and integrated.

    ## Analysis Steps:
    1. **Read the inputs carefully** and identify both the visual and textual input formats, understanding how they work together
    2. **Read all the assistant responses and corresponding feedback** to understand what went wrong with visual analysis, text processing, or their integration
    3. **Identify visual analysis patterns** - what visual features, relationships, or details are important for this task
    4. **Identify domain-specific knowledge** about both visual and textual aspects, as this information may not be available to the assistant in the future
    5. **Look for successful visual-textual integration strategies** and include these patterns in the instruction
    6. **Address specific visual analysis issues** mentioned in the feedback

    ## Instruction Requirements:
    - **Clear task definition** explaining how to process both visual and textual inputs
    - **Visual analysis guidance** specific to this task (what to look for, how to describe, what features matter)
    - **Integration strategies** for combining visual observations with textual information
    - **Domain-specific knowledge** about visual concepts, terminology, or relationships
    - **Error prevention guidance** for common visual analysis mistakes shown in the feedback
    - **Precise, actionable language** for both visual and textual processing

    Focus on creating an instruction that helps the assistant properly analyze visual content, integrate it with textual information, and avoid the specific visual analysis mistakes shown in the examples."""

    current_instruction = dspy.InputField(
        desc="The current instruction that was provided to the assistant to perform the multimodal task"
    )
    examples_with_feedback = dspy.InputField(
        desc="Task examples with visual content showing inputs, assistant outputs, and feedback. "
        "Pay special attention to feedback about visual analysis accuracy, visual-textual integration, "
        "and any domain-specific visual knowledge that the assistant missed."
    )

    improved_instruction = dspy.OutputField(
        desc="A better instruction for the assistant that addresses visual analysis issues, provides "
        "clear guidance on how to process and integrate visual and textual information, includes "
        "necessary visual domain knowledge, and prevents the visual analysis mistakes shown in the examples."
    )


class SingleComponentMultiModalProposer(dspy.Module):
    """
    dspy.Module for proposing improved instructions based on feedback.
    """

    def __init__(self):
        super().__init__()
        self.propose_instruction = dspy.Predict(GenerateEnhancedMultimodalInstructionFromFeedback)

    def forward(self, current_instruction: str, reflective_dataset: list[ReflectiveExample]) -> str:
        """
        Generate an improved instruction based on current instruction and feedback examples.

        Args:
            current_instruction: The current instruction that needs improvement
            reflective_dataset: List of examples with inputs, outputs, and feedback
                               May contain dspy.Image objects in inputs

        Returns:
            str: Improved instruction text
        """
        # Format examples with enhanced pattern recognition
        formatted_examples, image_map = self._format_examples_with_pattern_analysis(reflective_dataset)

        # Build kwargs for the prediction call
        predict_kwargs = {
            "current_instruction": current_instruction,
            "examples_with_feedback": formatted_examples,
        }

        # Create a rich multimodal examples_with_feedback that includes both text and images
        predict_kwargs["examples_with_feedback"] = self._create_multimodal_examples(formatted_examples, image_map)

        # Use current dspy LM settings (GEPA will pass reflection_lm via context)
        result = self.propose_instruction(**predict_kwargs)

        return result.improved_instruction

    def _format_examples_with_pattern_analysis(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples with pattern analysis and feedback categorization.

        Returns:
            tuple: (formatted_text_with_patterns, image_map)
        """
        # First, use the existing proven formatting approach
        formatted_examples, image_map = self._format_examples_for_instruction_generation(reflective_dataset)

        # Enhanced analysis: categorize feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(reflective_dataset)

        # Add pattern analysis to the formatted examples
        if feedback_analysis["summary"]:
            pattern_summary = self._create_pattern_summary(feedback_analysis)
            enhanced_examples = f"{pattern_summary}\n\n{formatted_examples}"
            return enhanced_examples, image_map

        return formatted_examples, image_map

    def _analyze_feedback_patterns(self, reflective_dataset: list[ReflectiveExample]) -> dict[str, Any]:
        """
        Analyze feedback patterns to provide better context for instruction generation.

        Categorizes feedback into:
        - Error patterns: Common mistakes and their types
        - Success patterns: What worked well and should be preserved/emphasized
        - Domain knowledge gaps: Missing information that should be included
        - Task-specific guidance: Specific requirements or edge cases
        """
        analysis = {
            "error_patterns": [],
            "success_patterns": [],
            "domain_knowledge_gaps": [],
            "task_specific_guidance": [],
            "summary": "",
        }

        # Simple pattern recognition - could be enhanced further
        for example in reflective_dataset:
            feedback = example.get("Feedback", "").lower()

            # Identify error patterns
            if any(error_word in feedback for error_word in ["incorrect", "wrong", "error", "failed", "missing"]):
                analysis["error_patterns"].append(feedback)

            # Identify success patterns
            if any(
                success_word in feedback for success_word in ["correct", "good", "accurate", "well", "successfully"]
            ):
                analysis["success_patterns"].append(feedback)

            # Identify domain knowledge needs
            if any(
                knowledge_word in feedback
                for knowledge_word in ["should know", "domain", "specific", "context", "background"]
            ):
                analysis["domain_knowledge_gaps"].append(feedback)

        # Create summary if patterns were found
        if any(analysis[key] for key in ["error_patterns", "success_patterns", "domain_knowledge_gaps"]):
            analysis["summary"] = (
                f"Patterns identified: {len(analysis['error_patterns'])} error(s), {len(analysis['success_patterns'])} success(es), {len(analysis['domain_knowledge_gaps'])} knowledge gap(s)"
            )

        return analysis

    def _create_pattern_summary(self, feedback_analysis: dict[str, Any]) -> str:
        """Create a summary of feedback patterns to help guide instruction generation."""

        summary_parts = ["## Feedback Pattern Analysis\n"]

        if feedback_analysis["error_patterns"]:
            summary_parts.append(f"**Common Issues Found ({len(feedback_analysis['error_patterns'])} examples):**")
            summary_parts.append("Focus on preventing these types of mistakes in the new instruction.\n")

        if feedback_analysis["success_patterns"]:
            summary_parts.append(
                f"**Successful Approaches Found ({len(feedback_analysis['success_patterns'])} examples):**"
            )
            summary_parts.append("Build on these successful strategies in the new instruction.\n")

        if feedback_analysis["domain_knowledge_gaps"]:
            summary_parts.append(
                f"**Domain Knowledge Needs Identified ({len(feedback_analysis['domain_knowledge_gaps'])} examples):**"
            )
            summary_parts.append("Include this specialized knowledge in the new instruction.\n")

        return "\n".join(summary_parts)

    def _format_examples_for_instruction_generation(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples using GEPA's markdown structure while preserving image objects.

        Returns:
            tuple: (formatted_text, image_map) where image_map maps example_index -> list[images]
        """

        def render_value_with_images(value, level=3, example_images=None):
            if example_images is None:
                example_images = []

            if isinstance(value, Type):
                image_idx = len(example_images) + 1
                example_images.append(value)
                return f"[IMAGE-{image_idx} - see visual content]\n\n"
            elif isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value_with_images(v, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value_with_images(item, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown_with_images(sample, example_num):
            example_images = []
            s = f"# Example {example_num}\n"

            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value_with_images(val, level=3, example_images=example_images)

            return s, example_images

        formatted_parts = []
        image_map = {}

        for i, example_data in enumerate(reflective_dataset):
            formatted_example, example_images = convert_sample_to_markdown_with_images(example_data, i + 1)
            formatted_parts.append(formatted_example)

            if example_images:
                image_map[i] = example_images

        formatted_text = "\n\n".join(formatted_parts)

        if image_map:
            total_images = sum(len(imgs) for imgs in image_map.values())
            formatted_text = (
                f"The examples below include visual content ({total_images} images total). "
                "Please analyze both the text and visual elements when suggesting improvements.\n\n" + formatted_text
            )

        return formatted_text, image_map

    def _create_multimodal_examples(self, formatted_text: str, image_map: dict[int, list[Type]]) -> Any:
        """
        Create a multimodal input that contains both text and images for the reflection LM.

        Args:
            formatted_text: The formatted text with image placeholders
            image_map: Dictionary mapping example_index -> list[images] for structured access
        """
        if not image_map:
            return formatted_text

        # Collect all images from all examples
        all_images = []
        for example_images in image_map.values():
            all_images.extend(example_images)

        multimodal_content = [formatted_text]
        multimodal_content.extend(all_images)
        return multimodal_content


class MultiModalInstructionProposer(ProposalFn):
    """GEPA-compatible multimodal instruction proposer.

    This class handles multimodal inputs (like dspy.Image) during GEPA optimization by using
    a single-component proposer for each component that needs to be updated.
    """

    def __init__(self):
        self.single_proposer = SingleComponentMultiModalProposer()

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """GEPA-compatible proposal function.

        Args:
            candidate: Current component name -> instruction mapping
            reflective_dataset: Component name -> list of reflective examples
            components_to_update: List of component names to update

        Returns:
            dict: Component name -> new instruction mapping
        """
        updated_components = {}

        for component_name in components_to_update:
            if component_name in candidate and component_name in reflective_dataset:
                current_instruction = candidate[component_name]
                component_reflective_data = reflective_dataset[component_name]

                # Call the single-instruction proposer.
                #
                # In the future, proposals could consider multiple components instructions,
                # instead of just the current instruction, for more holistic instruction proposals.
                new_instruction = self.single_proposer(
                    current_instruction=current_instruction, reflective_dataset=component_reflective_data
                )

                updated_components[component_name] = new_instruction

        return updated_components
