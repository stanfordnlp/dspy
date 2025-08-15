import json
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Dict

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class PlanAndExecute(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_plan_steps: int = 10, max_retries: int = 3, replan_enabled: bool = True):
        """
        PlanAndExecute is a framework for building agents that separate planning from execution.
        The agent first creates a comprehensive plan to accomplish the task, then executes each
        step sequentially using available tools. This approach provides more structured and
        deliberate task completion compared to ReAct's interleaved approach.

        Args:
            signature: The signature of the module, which defines the input and output.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_plan_steps (Optional[int]): The maximum number of steps in the plan. Defaults to 10.
            max_retries (Optional[int]): The maximum number of retries for failed steps. Defaults to 3.
            replan_enabled (Optional[bool]): Whether to enable dynamic replanning. Defaults to True.

        Example:

        ```python
        def list_files(directory: str) -> str:
            return "Found files: a.pdf, b.txt, c.docx"

        def get_file_details(filename: str) -> str:
            return f"Details of {filename}: size 1MB, created yesterday"

        plan_execute = dspy.PlanAndExecute(
            signature="request->response", 
            tools=[list_files, get_file_details],
            replan_enabled=True
        )
        
        # Example with replanning - the plan will be updated after step 1
        pred = plan_execute(request="Give me list of files and detail of each file")
        
        # Initial plan might be:
        # 1. {"id": 1, "description": "List all files", "replan": true}
        # 2. {"id": 2, "description": "Get details for each file found"}
        
        # After step 1 executes and finds files a.pdf, b.txt, c.docx,
        # the plan gets updated to:
        # 1. {"id": 1, "description": "List all files (completed)"}  
        # 2. {"id": 2, "description": "Get details of a.pdf"}
        # 3. {"id": 3, "description": "Get details of b.txt"}
        # 4. {"id": 4, "description": "Get details of c.docx"}
        # 5. {"id": 5, "description": "Aggregate all file details"}
        
        # Access the structured plan and execution results:
        print("Final plan:", pred.plan)  # Updated plan
        print("Execution history:", pred.execution_history)  # All executed steps
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_plan_steps = max_plan_steps
        self.max_retries = max_retries
        self.replan_enabled = replan_enabled

        # Convert tools to Tool objects and create a dictionary for lookup
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        self.tools = {tool.name: tool for tool in tools}
        self.tools["reasoning"] = Tool(
            func=lambda: "Completed.",
            name="reasoning", 
            desc="Use this when the step can be completed through reasoning alone without calling any external tools."
        )

        # Get input/output field names for instruction formatting
        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])

        # Build planning instructions
        plan_instr = self._build_planning_instructions(inputs, outputs)
        
        # Build execution instructions
        exec_instr = self._build_execution_instructions(outputs)
        
        # Build final answer extraction instructions
        extract_instr = self._build_extraction_instructions(inputs, outputs)

        # Create signatures for each phase
        self.planning_signature = (
            dspy.Signature({**signature.input_fields}, plan_instr)
            .append("plan", dspy.OutputField(desc="A JSON array of plan steps with id and description"), type_=str)
        )

        self.execution_signature = (
            dspy.Signature(
                {**signature.input_fields, "step_id": dspy.InputField(desc="ID of the current step to execute"), 
                 "step_description": dspy.InputField(desc="Description of the current step to execute"),
                 "plan": dspy.InputField(desc="The full plan in JSON format"), 
                 "execution_history": dspy.InputField(desc="History of executed steps and their results")},
                exec_instr
            )
            .append("step_id", dspy.OutputField(desc="ID of the current step to execute"), type_=str)
            .append("step_reasoning", dspy.OutputField(desc="Reasoning for this execution step"), type_=str)
            .append("tool_name", dspy.OutputField(desc="Name of the tool to use"), type_=str)
            .append("tool_args", dspy.OutputField(desc="Arguments for the tool in JSON format"), type_=dict[str, Any])
        )

        self.extraction_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields, 
             "plan": dspy.InputField(desc="The executed plan"), 
             "execution_history": dspy.InputField(desc="Complete execution history")},
            extract_instr,
        )

        # Create replanning signature if replan is enabled
        if self.replan_enabled:
            replan_instr = self._build_replanning_instructions(inputs, outputs)
            self.replanning_signature = (
                dspy.Signature({
                    **signature.input_fields,
                    "original_plan": dspy.InputField(desc="The original plan that is being updated"),
                    "execution_history": dspy.InputField(desc="History of executed steps and their results"),
                    "replan_step_result": dspy.InputField(desc="Result of the step that triggered replanning")
                }, replan_instr)
                .append("updated_plan", dspy.OutputField(desc="Updated plan incorporating new information"), type_=str)
            )

        # Create the sub-modules
        self.planner = dspy.ChainOfThought(self.planning_signature)
        self.executor = dspy.ChainOfThought(self.execution_signature)
        self.extractor = dspy.ChainOfThought(self.extraction_signature)
        if self.replan_enabled:
            self.replanner = dspy.ChainOfThought(self.replanning_signature)

    def _build_planning_instructions(self, inputs: str, outputs: str) -> str:
        """Build instructions for the planning phase."""
        base_instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        
        tools_desc = "\n".join([f"- {tool.name}: {tool.desc}" for tool in self.tools.values()])
        
        base_instr.extend([
            f"You are a strategic planning agent. Given the input fields {inputs}, create a comprehensive plan to produce {outputs}.",
            f"You have access to the following tools:\n{tools_desc}\n",
            f"Create a detailed step-by-step plan with at most {self.max_plan_steps} steps.",
            "Each step should be specific and actionable, clearly stating what tool to use and why.",
            "The plan should be logical, sequential, and comprehensive to fully accomplish the task.",
            "",
            "IMPORTANT: Output the plan as a JSON array where each step has an 'id' (integer starting from 1) and 'description' (string).",
            "Optionally, steps can include a 'replan' field (boolean) to indicate that the plan should be updated after this step completes.",
            "Use 'replan': true for steps where the results will determine what subsequent steps are needed.",
            "Example format:",
            '[',
            '  {"id": 1, "description": "Search for all files in the directory", "replan": true},',
            '  {"id": 2, "description": "Process each found file (this will be expanded after step 1)"}',
            ']',
            "When replan is true, the plan will be regenerated after that step to incorporate the step's results.",
            "Ensure the JSON is properly formatted and valid."
        ])
        
        return "\n".join(base_instr)

    def _build_execution_instructions(self, outputs: str) -> str:
        """Build instructions for the execution phase."""
        tools_list = "\n".join([f"({idx + 1}) {tool}" for idx, tool in enumerate(self.tools.values())])
        
        return "\n".join([
            "You are an execution agent following a predetermined plan.",
            "You are given a specific step ID and its description from the plan to execute.",
            "Execute the current step by selecting the appropriate tool and providing the correct arguments.",
            f"Available tools:\n{tools_list}\n",
            "Analyze the current step description, consider the execution history, and choose the right tool with proper arguments.",
            "IMPORTANT: If the current step can be completed through reasoning alone without external tools,",
            "use 'reasoning' and provide your reasoning in the 'step_reasoning' argument.",
            "Only use actual tools when external data or actions are genuinely needed.",
            "Provide clear reasoning for your tool selection and argument choices.",
            "The tool_args must be in valid JSON format that matches the tool's expected parameters."
        ])

    def _build_extraction_instructions(self, inputs: str, outputs: str) -> str:
        """Build instructions for the final answer extraction phase."""
        base_instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        
        base_instr.extend([
            f"You are a result extraction agent. Based on the input {inputs}, the executed plan, and the complete execution history,",
            f"extract and format the final answer to produce {outputs}.",
            "Synthesize information from all execution steps to provide a comprehensive and accurate response.",
            "Ensure the output directly addresses the original request and incorporates all relevant findings."
        ])
        
        return "\n".join(base_instr)

    def _build_replanning_instructions(self, inputs: str, outputs: str) -> str:
        """Build instructions for the replanning phase."""
        base_instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        
        tools_desc = "\n".join([f"- {tool.name}: {tool.desc}" for tool in self.tools.values()])
        
        base_instr.extend([
            f"You are a replanning agent. Based on the input {inputs}, the original plan, and the execution results so far,",
            f"create an updated plan to continue working toward producing {outputs}.",
            f"You have access to the following tools:\n{tools_desc}\n",
            "The original plan had a step marked for replanning, and that step has now been executed.",
            "Based on the results of that step, update the remaining plan to incorporate the new information.",
            "Keep completed steps as-is, but modify upcoming steps based on the new results.",
            "Ensure the updated plan will efficiently accomplish the remaining work.",
            "",
            "IMPORTANT: Output the updated plan as a JSON array where each step has an 'id', 'description', and optional 'replan' field.",
            "Maintain the same ID numbering for completed steps, and assign new IDs for new/modified steps.",
            "Example updated plan format:",
            '[',
            '  {"id": 1, "description": "Search for all files in the directory"},',
            '  {"id": 2, "description": "Get details of file1.pdf"},',
            '  {"id": 3, "description": "Get details of file2.txt"},',
            '  {"id": 4, "description": "Aggregate results"}',
            ']'
        ])
        
        return "\n".join(base_instr)

    def _format_execution_history(self, history: List[Dict[str, Any]]) -> str:
        """Format execution history for display."""
        if not history:
            return "No execution history yet."
        
        formatted = []
        for step in history:
            step_id = step.get('step_id', 'N/A')
            formatted.append(f"Step {step_id}:")
            formatted.append(f"  Description: {step.get('step_description', 'N/A')}")
            formatted.append(f"  Reasoning: {step.get('step_reasoning', 'N/A')}")
            formatted.append(f"  Tool: {step.get('tool_name', 'N/A')}")
            formatted.append(f"  Arguments: {step.get('tool_args', 'N/A')}")
            formatted.append(f"  Result: {step.get('result', 'N/A')}")
            formatted.append("")
        
        return "\n".join(formatted)

    def _parse_plan_steps(self, plan: str) -> List[Dict[str, Any]]:
        """Parse the JSON plan into individual steps."""
        try:
            # Try to parse as JSON
            plan_data = json.loads(plan.strip())
            
            # Ensure it's a list
            if not isinstance(plan_data, list):
                logger.warning("Plan is not a JSON array, attempting to extract steps")
                return []
            
            # Validate step structure and limit to max_plan_steps
            steps = []
            for item in plan_data[:self.max_plan_steps]:
                if isinstance(item, dict) and 'id' in item and 'description' in item:
                    step = { 'id': item['id'], 'description': item['description'] }
                    if 'replan' in item:
                        step['replan'] = bool(item['replan'])
                    steps.append(step)
                else:
                    logger.warning(f"Invalid step format: {item}")
            
            return steps
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan as JSON: {e}")
            # Fallback: try to extract steps from text format
            return self._fallback_parse_plan_steps(plan)
    
    def _fallback_parse_plan_steps(self, plan: str) -> List[Dict[str, Any]]:
        """Fallback parser for non-JSON plan format."""
        lines = plan.strip().split('\n')
        steps = []
        step_id = 1
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and bullet points
                step_text = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '•']:
                    if step_text.startswith(prefix):
                        step_text = step_text[len(prefix):].strip()
                        break
                if step_text:
                    steps.append({
                        'id': step_id,
                        'description': step_text
                    })
                    step_id += 1
        
        return steps[:self.max_plan_steps]

    def forward(self, **input_args):
        """Execute the plan-and-execute workflow."""
        max_plan_steps = input_args.pop("max_plan_steps", self.max_plan_steps)
        max_retries = input_args.pop("max_retries", self.max_retries)

        # Phase 1: Planning
        try:
            plan_result = self._call_with_potential_context_truncation(self.planner, {}, **input_args)
            plan = plan_result.plan
        except Exception as err:
            logger.error(f"Planning phase failed: {_fmt_exc(err)}")
            # Fallback to direct answer if planning fails
            return self._call_with_potential_context_truncation(
                self.extractor, 
                {"plan": "Planning failed", "execution_history": "No execution performed"},
                **input_args
            )

        # Parse plan into steps
        steps = self._parse_plan_steps(plan)
        if not steps:
            logger.warning("No valid steps found in plan")
            return self._call_with_potential_context_truncation(
                self.extractor,
                {"plan": plan, "execution_history": "No valid steps in plan"},
                **input_args
            )

        # Phase 2: Execution
        execution_history = []
        current_steps = steps.copy()  # Work with a copy that can be updated
        step_index = 0
        
        while step_index < len(current_steps):
            step = current_steps[step_index]
            step_id = step['id']
            step_description = step['description']
            step_executed = False
            
            for retry in range(max_retries):
                try:
                    # Execute current step
                    exec_result = self._call_with_potential_context_truncation(
                        self.executor,
                        {
                            "step_id": step_id,
                            "step_description": step_description,
                            "plan": plan,
                            "execution_history": self._format_execution_history(execution_history)
                        },
                        **input_args
                    )
                    
                    # Validate tool selection
                    if exec_result.tool_name not in self.tools:
                        raise ValueError(f"Invalid tool selected: {exec_result.tool_name}")
                    
                    # Execute the tool
                    if exec_result.tool_name == "no_tool":
                        tool_result = self.tools[exec_result.tool_name](reasoning=exec_result.step_reasoning)
                    else:
                        tool_result = self.tools[exec_result.tool_name](**exec_result.tool_args)
                    
                    # Record successful execution
                    execution_history.append({
                        "step_id": step_id,
                        "step_description": step_description,
                        "step_reasoning": exec_result.step_reasoning,
                        "tool_name": exec_result.tool_name,
                        "tool_args": exec_result.tool_args,
                        "result": tool_result,
                        "retry_count": retry
                    })
                    
                    step_executed = True
                    
                    # Check if this step requires replanning
                    if self.replan_enabled and step.get('replan', False):
                        logger.info(f"Step {step_id} marked for replanning, updating plan...")
                        try:
                            replan_result = self._call_with_potential_context_truncation(
                                self.replanner,
                                {
                                    "original_plan": json.dumps(current_steps, indent=2),
                                    "execution_history": self._format_execution_history(execution_history),
                                    "replan_step_result": str(tool_result)
                                },
                                **input_args
                            )
                            
                            # Parse the updated plan
                            new_steps = self._parse_plan_steps(replan_result.updated_plan)
                            if new_steps:
                                # Replace the current steps with the updated plan
                                current_steps = new_steps
                                logger.info(f"Plan updated with {len(new_steps)} steps")
                                plan = replan_result.updated_plan  # Update plan for final extraction
                            else:
                                logger.warning("Failed to parse updated plan, continuing with original plan")
                                
                        except Exception as replan_err:
                            logger.warning(f"Replanning failed: {_fmt_exc(replan_err)}, continuing with original plan")
                    
                    break
                    
                except Exception as err:
                    error_msg = f"Step {step_id} execution failed (attempt {retry + 1}): {_fmt_exc(err)}"
                    logger.warning(error_msg)
                    
                    if retry == max_retries - 1:
                        # Record failed execution after all retries
                        execution_history.append({
                            "step_id": step_id,
                            "step_description": step_description,
                            "step_reasoning": getattr(exec_result, 'step_reasoning', 'Failed to get reasoning'),
                            "tool_name": getattr(exec_result, 'tool_name', 'unknown'),
                            "tool_args": getattr(exec_result, 'tool_args', {}),
                            "result": f"FAILED: {error_msg}",
                            "retry_count": retry + 1
                        })
            
            # Move to next step
            step_index += 1

        # Phase 3: Extract final answer
        try:
            final_result = self._call_with_potential_context_truncation(
                self.extractor,
                {
                    "plan": plan,
                    "execution_history": self._format_execution_history(execution_history)
                },
                **input_args
            )
            
            # Add metadata to the result
            final_result.plan = steps
            final_result.execution_history = execution_history
            final_result.steps_executed = len([h for h in execution_history if not h["result"].startswith("FAILED:")])
            final_result.steps_failed = len([h for h in execution_history if h["result"].startswith("FAILED:")])
            
            return final_result
            
        except Exception as err:
            logger.error(f"Final extraction failed: {_fmt_exc(err)}")
            return dspy.Prediction(
                plan=steps,
                execution_history=execution_history,
                error=f"Extraction failed: {_fmt_exc(err)}"
            )

    async def aforward(self, **input_args):
        """Async version of the plan-and-execute workflow."""
        max_plan_steps = input_args.pop("max_plan_steps", self.max_plan_steps)
        max_retries = input_args.pop("max_retries", self.max_retries)

        # Phase 1: Planning
        try:
            plan_result = await self._async_call_with_potential_context_truncation(self.planner, {}, **input_args)
            plan = plan_result.plan
        except Exception as err:
            logger.error(f"Planning phase failed: {_fmt_exc(err)}")
            return await self._async_call_with_potential_context_truncation(
                self.extractor,
                {"plan": "Planning failed", "execution_history": "No execution performed"},
                **input_args
            )

        # Parse plan into steps
        steps = self._parse_plan_steps(plan)
        if not steps:
            logger.warning("No valid steps found in plan")
            return await self._async_call_with_potential_context_truncation(
                self.extractor,
                {"plan": plan, "execution_history": "No valid steps in plan"},
                **input_args
            )

        # Phase 2: Execution
        execution_history = []
        current_steps = steps.copy()  # Work with a copy that can be updated
        step_index = 0
        
        while step_index < len(current_steps):
            step = current_steps[step_index]
            step_id = step['id']
            step_description = step['description']
            step_executed = False
            
            for retry in range(max_retries):
                try:
                    # Execute current step
                    exec_result = await self._async_call_with_potential_context_truncation(
                        self.executor,
                        {
                            "step_id": step_id,
                            "step_description": step_description,
                            "plan": plan,
                            "execution_history": self._format_execution_history(execution_history)
                        },
                        **input_args
                    )
                    
                    # Validate tool selection
                    if exec_result.tool_name not in self.tools:
                        raise ValueError(f"Invalid tool selected: {exec_result.tool_name}")
                    
                    # Execute the tool (async if available)
                    if exec_result.tool_name == "no_tool":
                        # For no_tool, pass the reasoning from the execution result
                        tool_result = self.tools[exec_result.tool_name](reasoning=exec_result.step_reasoning)
                    elif hasattr(self.tools[exec_result.tool_name], 'acall'):
                        tool_result = await self.tools[exec_result.tool_name].acall(**exec_result.tool_args)
                    else:
                        tool_result = self.tools[exec_result.tool_name](**exec_result.tool_args)
                    
                    # Record successful execution
                    execution_history.append({
                        "step_id": step_id,
                        "step_description": step_description,
                        "step_reasoning": exec_result.step_reasoning,
                        "tool_name": exec_result.tool_name,
                        "tool_args": exec_result.tool_args,
                        "result": tool_result,
                        "retry_count": retry
                    })
                    
                    step_executed = True
                    
                    # Check if this step requires replanning
                    if self.replan_enabled and step.get('replan', False):
                        logger.info(f"Step {step_id} marked for replanning, updating plan...")
                        try:
                            replan_result = await self._async_call_with_potential_context_truncation(
                                self.replanner,
                                {
                                    "original_plan": json.dumps(current_steps, indent=2),
                                    "execution_history": self._format_execution_history(execution_history),
                                    "replan_step_result": str(tool_result)
                                },
                                **input_args
                            )
                            
                            # Parse the updated plan
                            new_steps = self._parse_plan_steps(replan_result.updated_plan)
                            if new_steps:
                                # Replace the current steps with the updated plan
                                current_steps = new_steps
                                logger.info(f"Plan updated with {len(new_steps)} steps")
                                plan = replan_result.updated_plan  # Update plan for final extraction
                            else:
                                logger.warning("Failed to parse updated plan, continuing with original plan")
                                
                        except Exception as replan_err:
                            logger.warning(f"Replanning failed: {_fmt_exc(replan_err)}, continuing with original plan")
                    
                    break
                    
                except Exception as err:
                    error_msg = f"Step {step_id} execution failed (attempt {retry + 1}): {_fmt_exc(err)}"
                    logger.warning(error_msg)
                    
                    if retry == max_retries - 1:
                        # Record failed execution after all retries
                        execution_history.append({
                            "step_id": step_id,
                            "step_description": step_description,
                            "step_reasoning": getattr(exec_result, 'step_reasoning', 'Failed to get reasoning'),
                            "tool_name": getattr(exec_result, 'tool_name', 'unknown'),
                            "tool_args": getattr(exec_result, 'tool_args', {}),
                            "result": f"FAILED: {error_msg}",
                            "retry_count": retry + 1
                        })
            
            # Move to next step
            step_index += 1

        # Phase 3: Extract final answer
        try:
            final_result = await self._async_call_with_potential_context_truncation(
                self.extractor,
                {
                    "plan": plan,
                    "execution_history": self._format_execution_history(execution_history)
                },
                **input_args
            )
            
            # Add metadata to the result
            final_result.plan = steps
            final_result.execution_history = execution_history
            final_result.steps_executed = len([h for h in execution_history if not h["result"].startswith("FAILED:")])
            final_result.steps_failed = len([h for h in execution_history if h["result"].startswith("FAILED:")])
            
            return final_result
            
        except Exception as err:
            logger.error(f"Final extraction failed: {_fmt_exc(err)}")
            return dspy.Prediction(
                plan=steps,
                execution_history=execution_history,
                error=f"Extraction failed: {_fmt_exc(err)}"
            )

    def _call_with_potential_context_truncation(self, module, additional_args, **input_args):
        """Call a module with potential context window truncation handling."""
        for _ in range(3):
            try:
                return module(**input_args, **additional_args)
            except ContextWindowExceededError:
                logger.warning("Context window exceeded, truncating execution history.")
                if "execution_history" in additional_args:
                    additional_args["execution_history"] = self._truncate_execution_history(
                        additional_args["execution_history"]
                    )
                else:
                    # If no execution history to truncate, raise the error
                    raise

    async def _async_call_with_potential_context_truncation(self, module, additional_args, **input_args):
        """Async version of context truncation handling."""
        for _ in range(3):
            try:
                return await module.acall(**input_args, **additional_args)
            except ContextWindowExceededError:
                logger.warning("Context window exceeded, truncating execution history.")
                if "execution_history" in additional_args:
                    additional_args["execution_history"] = self._truncate_execution_history(
                        additional_args["execution_history"]
                    )
                else:
                    # If no execution history to truncate, raise the error
                    raise

    def _truncate_execution_history(self, history: str) -> str:
        """Truncate execution history to fit in context window."""
        lines = history.split('\n')
        if len(lines) <= 10:
            return "Execution history truncated due to context limits."
        
        # Keep the first few and last few lines
        truncated = lines[:5] + ["... (history truncated) ..."] + lines[-5:]
        return '\n'.join(truncated)


def _fmt_exc(err: BaseException, *, limit: int = 3) -> str:
    """Return a one-string traceback summary."""
    import traceback
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Design Notes and Implementation Details for dspy.PlanAndExecute

ARCHITECTURAL OVERVIEW:
The PlanAndExecute module implements the Plan-and-Execute agent framework, which separates planning 
from execution in a structured three-phase approach:

1. Planning Phase: Creates a comprehensive step-by-step plan to accomplish the task
2. Execution Phase: Sequentially executes each planned step using available tools
3. Extraction Phase: Synthesizes results from execution into the final answer

DIFFERENCES FROM ReAct:
- ReAct: Interleaved thinking → acting → observing in a loop
- PlanAndExecute: Separated planning → execution → extraction phases
- More structured and deliberate approach vs ReAct's flexible iteration
- Better for complex multi-step tasks that benefit from upfront planning

KEY FEATURES:
- Three separate DSPy signatures for planning, execution, and extraction phases
- JSON-structured plans with step IDs for better traceability and organization
- Robust error handling with configurable retry logic for failed steps
- Context window management with automatic truncation of execution history
- Full async support for both modules and tool execution
- Comprehensive execution history tracking with step ID references
- Compatible with DSPy streaming and callback systems
- Fallback parsing for non-JSON plan formats

STREAMING SUPPORT:
The module fully supports DSPy's streaming functionality through the streamify wrapper.
Each phase (planning, execution steps, extraction) can be streamed independently.

TOOL INTEGRATION:
- Uses the same Tool abstraction as ReAct for consistency
- Supports both sync and async tool execution
- Tools are validated at execution time with clear error messages
- No special "finish" tool needed (unlike ReAct) as execution follows the predetermined plan
- Built-in "no_tool" option for steps that can be completed through reasoning alone
- Automatically handles mixed workflows combining tool usage and pure reasoning steps

CONTEXT MANAGEMENT:
- Implements truncation strategies for execution history when context limits are reached
- Preserves essential information while managing token usage
- Graceful degradation when context cannot be truncated further

FUTURE IMPROVEMENTS:
- Add support for dynamic plan modification based on execution results
- Implement plan validation and feasibility checking
- Support for conditional execution and branching in plans
- Integration with reflection modules for plan quality assessment
- Memory persistence across multiple forward calls for complex workflows

USAGE PATTERNS:
- Best suited for complex tasks requiring structured decomposition
- Excellent for workflows with clear sequential dependencies
- Ideal when planning overhead is justified by task complexity
- Consider ReAct for more exploratory or reactive task patterns
"""
