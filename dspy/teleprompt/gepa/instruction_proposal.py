import json
import logging
from typing import Any

from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.types.base_type import Type
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample

logger = logging.getLogger(__name__)


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


class GenerateImprovedToolModuleDescriptionsFromFeedback(dspy.Signature):
    """I provided an assistant with predictor instructions and tool descriptions,
    but its performance needs improvement based on the examples_with_feedback below.

    Your task is to propose better predictor instructions, tool descriptions, and tool argument descriptions that address the issues shown in these examples.
    Focus on reinforcing patterns that clearly improve the assistant's performance on similar tasks, rather than rewriting everything from scratch unless necessary.
    These components are progressively optimized - refine only what needs to change.

    Analyze the examples_with_feedback to identify success and failure patterns, and write improved instructions and descriptions at their appropriate level of abstraction and/or specificity,
    so that each layer plays a clear, complementary role without unnecessary repetition or verbosity unless redundancy clearly helps the assistant's performance.
    """

    current_predictor_instruction = dspy.InputField(desc="Current instruction guiding the predictor")
    current_tools = dspy.InputField(annotation=list[dspy.Tool], desc="Available tools with their complete schemas")
    examples_with_feedback = dspy.InputField(desc="Execution examples with feedback showing successes and failures")

    improved_predictor_instruction: str | None = dspy.OutputField(
        desc="Improved instruction for the predictor", default=None
    )


class ToolProposer(ProposalFn):
    """Proposer for optimizing tool-using module configurations.

    Supports two types of modules:
    - Tool modules (1 predictor): Optimizes predictor instruction and tool descriptions
    - ReAct modules (2 predictors): Jointly optimizes react instruction, extract instruction, and tool descriptions

    Uses dynamic signature generation to create output fields for each tool and parameter,
    enabling the reflection LM to optimize all components cohesively based on execution feedback.

    This joint optimization approach allows the LM to see how instructions and tool descriptions
    work together, leading to more coherent improvements than optimizing each component separately.
    """

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Optimize tool-using module components.

        Args:
            candidate: Current component name -> JSON config mapping
            reflective_dataset: Component name -> list of reflective examples
            components_to_update: List of tool-using module component names to update

        Returns:
            dict: Mapping of component names to improved JSON configs
        """

        updated_components = {}

        for module_key in components_to_update:
            if module_key not in candidate or module_key not in reflective_dataset:
                logger.debug(
                    f"Skipping {module_key}: not in candidate={module_key not in candidate}, not in "
                    "reflective_dataset={module_key not in reflective_dataset}"
                )
                continue
            current_module_config = json.loads(candidate[module_key])

            # Predictor keys: ReAct has 2 predictors (react + extract)
            predictor_keys = [k for k, v in current_module_config.items() if isinstance(v, str)]
            primary_predictor_key = predictor_keys[0]
            extract_predictor_key = predictor_keys[1] if len(predictor_keys) > 1 else None

            # Reconstruct Tool objects from JSON (func is placeholder since it can't be serialized)
            current_tools_dict = current_module_config.get("tools", {})
            tools_list = []
            for tool_name, tool_info in current_tools_dict.items():
                tool = dspy.Tool(
                    func=lambda *args, **kwargs: None,  # Placeholder - Tool requires Callable, but only schema is used
                    name=tool_name,
                    desc=tool_info.get("desc", ""),
                )
                tool.args = tool_info.get("args", {})
                tools_list.append(tool)

            # Build dynamic signature with tool-specific output fields
            signature = GenerateImprovedToolModuleDescriptionsFromFeedback

            for tool in tools_list:
                tool_name = tool.name
                tool_info = current_tools_dict[tool_name]

                signature = signature.append(
                    f"improved_tool_{tool_name}_desc",
                    dspy.OutputField(desc=f"Improved description of tool '{tool_name}'", default=None),
                )

                for arg_name in tool_info["args"].keys():
                    signature = signature.append(
                        f"improved_tool_{tool_name}_arg_{arg_name}_desc",
                        dspy.OutputField(
                            desc=f"Improved description of the argument '{arg_name}' of tool '{tool_name}'",
                            default=None,
                        ),
                    )

            kwargs = {
                "current_predictor_instruction": current_module_config[primary_predictor_key],
                "current_tools": tools_list,
                "examples_with_feedback": self._format_examples(reflective_dataset[module_key]),
            }
            # If module has extract predictor, add extract fields
            if extract_predictor_key is not None:
                signature = signature.append(
                    "current_extract_instruction", dspy.InputField(desc="Current instruction for extraction predictor")
                )
                signature = signature.append(
                    "improved_extract_instruction",
                    dspy.OutputField(desc="Improved instruction for extraction", default=None),
                )
                kwargs["current_extract_instruction"] = current_module_config[extract_predictor_key]

            propose_descriptions = dspy.Predict(signature)
            result = propose_descriptions(**kwargs)

            # Build improved config (reflection LM returns None to keep original, or new text)
            improved_module_config = {}

            if result.improved_predictor_instruction is not None:
                improved_module_config[primary_predictor_key] = result.improved_predictor_instruction

            if extract_predictor_key is not None and result.improved_extract_instruction is not None:
                improved_module_config[extract_predictor_key] = result.improved_extract_instruction

            improved_module_config["tools"] = {}
            for tool_name, tool_info in current_tools_dict.items():
                # Update tool description if LM proposed a change
                improved_tool_desc = getattr(result, f"improved_tool_{tool_name}_desc", None)
                if improved_tool_desc is not None:
                    tool_info["desc"] = improved_tool_desc

                # Update arg descriptions if LM proposed changes
                for arg_name in tool_info["args"].keys():
                    improved_tool_arg_desc = getattr(result, f"improved_tool_{tool_name}_arg_{arg_name}_desc", None)
                    if improved_tool_arg_desc is not None:
                        tool_info["args"][arg_name]["description"] = improved_tool_arg_desc

                improved_module_config["tools"][tool_name] = tool_info

            updated_components[module_key] = json.dumps(improved_module_config, indent=2)

        return updated_components

    def _format_examples(self, reflective_dataset: list[ReflectiveExample]) -> str:
        """Format reflective examples using GEPA's markdown structure."""

        def render_value(value, level=3):
            if isinstance(value, dict):
                s = ""
                for key, val in value.items():
                    s += f"{'#' * level} {key}\n"
                    s += render_value(val, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            if isinstance(value, (list, tuple)):
                s = ""
                for index, item in enumerate(value):
                    s += f"{'#' * level} Item {index + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown(sample, example_num):
            s = f"# Example {example_num}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value(val, level=3)
            return s

        formatted_parts = [convert_sample_to_markdown(example, i + 1) for i, example in enumerate(reflective_dataset)]
        return "\n\n".join(formatted_parts)
