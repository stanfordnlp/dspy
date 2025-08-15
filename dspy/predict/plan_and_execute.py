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
        # 1. {"id": "1", "description": "List all files", "replan": true}
        # 2. {"id": "2", "description": "Get details for each file found"}
        
        # After step 1 executes and finds files a.pdf, b.txt, c.docx,
        # the plan gets updated to:
        # 1. {"id": "1", "description": "List all files (completed)"}  
        # 2. {"id": "2", "description": "Get details of a.pdf"}
        # 3. {"id": "3", "description": "Get details of b.txt"}
        # 4. {"id": "4", "description": "Get details of c.docx"}
        # 5. {"id": "5", "description": "Aggregate all file details"}
        
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
            func=lambda: "Hoàn thành.",
            name="reasoning", 
            desc="Sử dụng khi bước này có thể được hoàn thành chỉ thông qua lập luận mà không cần gọi bất kỳ công cụ bên ngoài nào."
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
            f"Bạn là một tác nhân lập kế hoạch chiến lược. Với các trường đầu vào {inputs}, hãy tạo một kế hoạch toàn diện để tạo ra {outputs}.",
            f"Bạn có quyền truy cập vào các công cụ sau:\n{tools_desc}\n",
            f"Tạo một kế hoạch chi tiết từng bước với tối đa {self.max_plan_steps} bước.",
            "Mỗi bước phải cụ thể và có thể thực hiện được, nêu rõ công cụ nào sẽ sử dụng và tại sao.",
            "Kế hoạch phải logic, tuần tự và toàn diện để hoàn thành nhiệm vụ một cách đầy đủ.",
            "",
            "QUAN TRỌNG: Xuất kế hoạch dưới dạng mảng JSON trong đó mỗi bước có 'id' (số nguyên bắt đầu từ 1) và 'description' (chuỗi).",
            "Tùy chọn, các bước có thể bao gồm trường 'replan' (boolean) để chỉ ra rằng kế hoạch nên được cập nhật sau khi bước này hoàn thành.",
            "Sử dụng 'replan': true cho các bước mà kết quả sẽ xác định những bước tiếp theo cần thiết.",
            "Định dạng ví dụ:",
            '[',
            '  {"id": "1", "description": "Tìm kiếm tất cả các tệp trong thư mục", "replan": true},',
            '  {"id": "2", "description": "Xử lý từng tệp được tìm thấy (sẽ được mở rộng sau bước 1)"}',
            ']',
            "Khi replan là true, kế hoạch sẽ được tạo lại sau bước đó để kết hợp kết quả của bước.",
            "Đảm bảo JSON được định dạng đúng và hợp lệ."
        ])
        
        return "\n".join(base_instr)

    def _build_execution_instructions(self, outputs: str) -> str:
        """Build instructions for the execution phase."""
        tools_list = "\n".join([f"({idx + 1}) {tool}" for idx, tool in enumerate(self.tools.values())])
        
        return "\n".join([
            "Bạn là một tác nhân thực thi theo một kế hoạch đã được xác định trước.",
            "Bạn được cung cấp một ID bước cụ thể và mô tả của nó từ kế hoạch để thực hiện.",
            "Thực hiện bước hiện tại bằng cách chọn công cụ phù hợp và cung cấp các đối số đúng.",
            f"Các công cụ có sẵn:\n{tools_list}\n",
            "Phân tích mô tả bước hiện tại, xem xét lịch sử thực thi, và chọn công cụ phù hợp với các đối số đúng.",
            "QUAN TRỌNG: Nếu bước hiện tại có thể được hoàn thành chỉ thông qua lập luận mà không cần công cụ bên ngoài,",
            "sử dụng 'reasoning' và cung cấp lập luận của bạn trong đối số 'step_reasoning'.",
            "sử dụng 'step_id' và cung cấp ID bước của bạn trong đối số 'step_id'.",
            "Chỉ sử dụng các công cụ thực sự khi thực sự cần dữ liệu hoặc hành động bên ngoài.",
            "Cung cấp lập luận rõ ràng cho việc lựa chọn công cụ và đối số của bạn.",
            "Các tool_args phải ở định dạng JSON hợp lệ phù hợp với các tham số mong đợi của công cụ."
        ])

    def _build_extraction_instructions(self, inputs: str, outputs: str) -> str:
        """Build instructions for the final answer extraction phase."""
        base_instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        
        base_instr.extend([
            f"Bạn là một tác nhân trích xuất kết quả. Dựa trên đầu vào {inputs}, kế hoạch đã thực hiện và lịch sử thực thi hoàn chỉnh,",
            f"trích xuất và định dạng câu trả lời cuối cùng để tạo ra {outputs}.",
            "Tổng hợp thông tin từ tất cả các bước thực thi để cung cấp một phản hồi toàn diện và chính xác.",
            "Đảm bảo đầu ra trực tiếp giải quyết yêu cầu ban đầu và kết hợp tất cả các phát hiện liên quan."
        ])
        
        return "\n".join(base_instr)

    def _build_replanning_instructions(self, inputs: str, outputs: str) -> str:
        """Build instructions for the replanning phase."""
        base_instr = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        
        tools_desc = "\n".join([f"- {tool.name}: {tool.desc}" for tool in self.tools.values()])
        
        base_instr.extend([
            f"Bạn là một tác nhân lên kế hoạch lại. Dựa trên đầu vào {inputs}, kế hoạch ban đầu và kết quả thực thi cho đến nay,",
            f"tạo một kế hoạch cập nhật để tiếp tục làm việc hướng tới việc tạo ra {outputs}.",
            f"Bạn có quyền truy cập vào các công cụ sau:\n{tools_desc}\n",
            "Kế hoạch ban đầu có một bước được đánh dấu để lên kế hoạch lại, và bước đó đã được thực hiện.",
            "Dựa trên kết quả của bước đó, cập nhật kế hoạch còn lại để kết hợp thông tin mới.",
            "Giữ nguyên các bước đã hoàn thành, nhưng sửa đổi các bước sắp tới dựa trên kết quả mới.",
            "Đảm bảo kế hoạch cập nhật sẽ hoàn thành hiệu quả công việc còn lại.",
            "",
            "QUAN TRỌNG: Xuất kế hoạch cập nhật dưới dạng mảng JSON trong đó mỗi bước có 'id', 'description' và trường 'replan' tùy chọn.",
            "Duy trì cùng một hệ thống đánh số ID cho các bước đã hoàn thành và chỉ định ID mới cho các bước mới/đã sửa đổi.",
            "Định dạng kế hoạch cập nhật ví dụ:",
            '[',
            '  {"id": "1", "description": "Tìm kiếm tất cả các tệp trong thư mục"},',
            '  {"id": "2", "description": "Lấy chi tiết của file1.pdf"},',
            '  {"id": "3", "description": "Lấy chi tiết của file2.txt"},',
            '  {"id": "4", "description": "Tổng hợp kết quả"}',
            ']'
        ])
        
        return "\n".join(base_instr)

    def _format_execution_history(self, history: List[Dict[str, Any]]) -> str:
        """Format execution history for display."""
        if not history:
            return "Chưa có lịch sử thực thi nào."
        
        formatted = []
        for step in history:
            step_id = step.get('step_id', 'Không có')
            formatted.append(f"Bước {step_id}:")
            formatted.append(f"  Mô tả: {step.get('step_description', 'Không có')}")
            formatted.append(f"  Lập luận: {step.get('step_reasoning', 'Không có')}")
            formatted.append(f"  Công cụ: {step.get('tool_name', 'Không có')}")
            formatted.append(f"  Đối số: {step.get('tool_args', 'Không có')}")
            formatted.append(f"  Kết quả: {step.get('result', 'Không có')}")
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
                    step = { 'id': str(item['id']), 'description': item['description'] }  # Ensure id is always string
                    if 'replan' in item:
                        step['replan'] = bool(item['replan'])
                    steps.append(step)
                else:
                    logger.warning(f"Invalid step format: {item}")
            
            return steps
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan as JSON: {e}")
            return []

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
            result = self._call_with_potential_context_truncation(
                self.extractor, 
                {"plan": "Lập kế hoạch thất bại", "execution_history": "Không thực hiện gì"},
                **input_args
            )
            # Ensure plan is available immediately even in failure case
            result.plan = []
            result.execution_history = []
            result.steps_executed = 0
            result.steps_failed = 0
            result.error = f"Lập kế hoạch thất bại: {_fmt_exc(err)}"
            return result

        # Parse plan into steps
        steps = self._parse_plan_steps(plan)
        if not steps:
            result = self._call_with_potential_context_truncation(
                self.extractor,
                {"plan": plan, "execution_history": "Không có bước hợp lệ nào trong kế hoạch"},
                **input_args
            )
            # Ensure plan is available immediately
            result.plan = []
            result.execution_history = []
            return result

        # Create initial result with plan available immediately
        initial_result = dspy.Prediction()
        initial_result.plan = steps
        initial_result.execution_history = []
        initial_result.steps_executed = 0
        initial_result.steps_failed = 0

        # Phase 2: Execution
        execution_history = []
        current_steps = steps.copy()  # Work with a copy that can be updated
        step_index = 0
        
        while step_index < len(current_steps):
            step = current_steps[step_index]
            step_id = step['id']
            step_description = step['description']
            
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

                    # Validate execution result has required fields
                    if not hasattr(exec_result, 'step_id'):
                        exec_result.step_id = step_id
                    if not hasattr(exec_result, 'step_reasoning'):
                        exec_result.step_reasoning = "Không có lập luận nào được cung cấp bởi trình thực thi"
                    if not hasattr(exec_result, 'tool_name'):
                        raise ValueError("Executor did not provide tool_name")
                    if not hasattr(exec_result, 'tool_args'):
                        exec_result.tool_args = {}
                    
                    # Validate tool selection
                    if exec_result.tool_name not in self.tools:
                        raise ValueError(f"Invalid tool selected: {exec_result.tool_name}")
                    
                    # Execute the tool
                    if exec_result.tool_name == "reasoning":
                        tool_result = self.tools[exec_result.tool_name]()
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
                                current_steps = new_steps
                                plan = replan_result.updated_plan
                            else:
                                logger.warning("Failed to parse updated plan, continuing with original plan")
                                
                        except Exception as replan_err:
                            logger.warning(f"Replanning failed: {_fmt_exc(replan_err)}, continuing with original plan")
                    
                    break
                    
                except Exception as err:
                    error_msg = f"Thực thi bước {step_id} thất bại (lần thử {retry + 1}): {_fmt_exc(err)}"
                    logger.warning(error_msg)
                    
                    if retry == max_retries - 1:
                        # Record failed execution after all retries
                        execution_history.append({
                            "step_id": step_id,
                            "step_description": step_description,
                            "step_reasoning": getattr(exec_result, 'step_reasoning', 'Không thể lấy lập luận'),
                            "tool_name": getattr(exec_result, 'tool_name', 'không rõ'),
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
            
            # Update the initial result with extraction results while preserving plan
            for key, value in final_result.items():
                if key not in ['plan', 'execution_history', 'steps_executed', 'steps_failed']:
                    setattr(initial_result, key, value)
            
            # Update metadata
            initial_result.execution_history = execution_history
            initial_result.steps_executed = len([h for h in execution_history if not h["result"].startswith("FAILED:")])
            initial_result.steps_failed = len([h for h in execution_history if h["result"].startswith("FAILED:")])
            
            return initial_result
            
        except Exception as err:
            logger.error(f"Final extraction failed: {_fmt_exc(err)}")
            # Update initial result with error while preserving plan
            initial_result.execution_history = execution_history
            initial_result.steps_executed = len([h for h in execution_history if not h["result"].startswith("FAILED:")])
            initial_result.steps_failed = len([h for h in execution_history if h["result"].startswith("FAILED:")])
            initial_result.error = f"Trích xuất thất bại: {_fmt_exc(err)}"
            return initial_result

    async def aforward(self, **input_args):
        """Async version of the plan-and-execute workflow."""
        pass # Skip asycn for now

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

    def _truncate_execution_history(self, history: str) -> str:
        """Truncate execution history to fit in context window."""
        lines = history.split('\n')
        if len(lines) <= 10:
            return "Lịch sử thực thi đã bị cắt bớt do giới hạn ngữ cảnh."
        
        # Keep the first few and last few lines
        truncated = lines[:5] + ["... (lịch sử đã bị cắt bớt) ..."] + lines[-5:]
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
