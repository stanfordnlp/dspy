import dspy
import re
import json
import asyncio
import concurrent.futures
from pydantic import BaseModel, Field
from typing import List, Tuple, Any, Optional, Callable, Dict, Union
import time

# Define Pydantic model for state evaluation output
class StateEvaluationOutput(BaseModel):
    """Pydantic model for validating state evaluation step output."""
    score: float = Field(..., description="Evaluation score for the current path's prospect, between 0.0 and 1.0.")
    reasoning: str = Field(..., description="Brief explanation or reasoning for the evaluation score.")

class TreeNode:
    """Represents a node in the tree of thought"""
    def __init__(self, thought: str = "", parent=None, score: float = 0.0, reasoning: str = "", depth: int = 0):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.score = score
        self.reasoning = reasoning
        self.depth = depth
        self.is_best_path = False
    
    def add_child(self, thought: str, score: float = 0.0, reasoning: str = ""):
        """Add a child node and return the newly created node"""
        child = TreeNode(thought, self, score, reasoning, self.depth + 1)
        self.children.append(child)
        return child
    
    def path_to_root(self) -> List[str]:
        """Get path from root node to current node"""
        if self.parent is None:
            return [self.thought] if self.thought else []
        else:
            return self.parent.path_to_root() + [self.thought]

class TreeOfThought(dspy.Module):
    def __init__(self, signature, max_depth: int = 3, num_branches: int = 3, beam_width: int = 3, 
                 metric: Optional[Callable] = None, parallel: bool = True, max_workers: int = 4, **config):
        """
        A DSPy module that explores problem-solving using tree of thought approach with beam search.

        Args:
            signature: The dspy.Signature describing the overall task (e.g., "question -> answer").
            max_depth (int): Maximum depth of the thinking tree.
            num_branches (int): Number of thoughts (branches) to generate at each step (k).
            beam_width (int): Number of best states to keep at each step (beam search width).
            metric (Callable, optional): Function to evaluate state prospects.
                                       If None, uses an LLM-based evaluator.
                                       Signature: metric(state: List[str], **kwargs) -> float
            parallel (bool): Whether to enable parallel evaluation.
            max_workers (int): Maximum number of worker threads for parallel evaluation.
            **config: Additional configuration passed to internal predictors.
        """
        super().__init__()
        self.signature = dspy.ensure_signature(signature)
        self.max_depth = max_depth
        self.num_branches = num_branches
        self.beam_width = beam_width
        self.metric = metric
        self.config = config
        
        # Parallel evaluation configuration
        self.parallel = parallel
        self.max_workers = max_workers
        
        # Internal state
        self.root_node = TreeNode(thought="", depth=0)
        self.best_paths = []
        
        # --- Internal predictors ---

        # 1. Generate multiple next thoughts (using ChainOfThought)
        thought_gen_base_sig_fields = {
            "problem_description": dspy.InputField(desc="The overall problem to solve."),
            "current_path": dspy.InputField(desc="Sequence of thoughts generated so far."),
            "next_thoughts": dspy.OutputField(desc=f"Generate {self.num_branches} different, potential next steps or thoughts to continue the reasoning path. Output should be a clearly numbered or separated list.")
        }
        self.generate_thoughts = dspy.ChainOfThought(
            dspy.Signature(thought_gen_base_sig_fields, "Think step by step to generate potential next thoughts."),
            **config
        )

        # 2. Evaluate state prospects (if no external metric provided)
        self.evaluate_state = None
        if self.metric is None:
            json_format_example = """```json {"score": <float, e.g., 0.75>, "reasoning": "<string, Your reasoning here>"} ```"""
            eval_base_sig_fields = {
                "problem_description": dspy.InputField(desc="The overall problem to solve."),
                "current_path": dspy.InputField(desc="Sequence of thoughts generated so far."),
                "evaluation_results": dspy.OutputField(
                    desc=f"Provide evaluation score and reasoning as a JSON object. JSON object must contain 'score' (float, between 0.0 and 1.0) and 'reasoning' (string) keys. **Output strictly in the format specified below, only output the JSON code block:** {json_format_example} Ensure the JSON format is correct.",
                    type_=StateEvaluationOutput
                )
            }
            eval_instructions = ("Evaluate the prospect of the current reasoning path. Provide a numeric score between 0.0 and 1.0 "
                              "indicating the likelihood that this path leads to a correct solution, along with your reasoning. "
                              "**Output must be a single, valid JSON code block in the format described above.**")
            self.evaluate_state = dspy.Predict(
                dspy.Signature(eval_base_sig_fields, eval_instructions),
                **config
            )

        # 3. Generate final answer based on best path (using ChainOfThought)
        final_answer_base_sig_fields = {
            "problem_description": dspy.InputField(desc="The overall problem to solve."),
            "best_path": dspy.InputField(desc="The most promising sequence of thoughts leading to a solution."),
            **self.signature.output_fields
        }
        output_field_names = ', '.join(self.signature.output_fields.keys())
        self.generate_final_answer = dspy.ChainOfThought(
            dspy.Signature(final_answer_base_sig_fields, f"Based on the problem and the best reasoning path, think step by step to generate the final output fields: {output_field_names}."),
            **config
        )

    def _extract_and_parse_json(self, response_str: str) -> Tuple[float, str]:
        """
        Extract and parse JSON from LLM response, compatible with both ChatAdapter and JSONAdapter formats.
        
        Args:
            response_str: LLM response string
            
        Returns:
            Tuple[float, str]: Parsed score and reasoning
        """
        # 1. Try ChatAdapter format: extract content between [[ ## evaluation_results ## ]] markers
        chat_pattern = r'\[\[ ## evaluation_results ## \]\]\s*(.*?)(?:\[\[ ## completed ## \]|$)'
        chat_match = re.search(chat_pattern, response_str, re.DOTALL)
        
        # 2. Try to extract JSON code block
        markdown_pattern = r'```(?:json)?\s*(.*?)```'
        markdown_match = re.search(markdown_pattern, response_str, re.DOTALL)
        
        # 3. Try to extract JSON object directly from the entire string
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        json_match = re.search(json_pattern, response_str, re.DOTALL)
        
        # Try different extraction results by priority
        extracted_str = None
        if chat_match:
            extracted_str = chat_match.group(1).strip()
        elif markdown_match:
            extracted_str = markdown_match.group(1).strip()
        elif json_match:
            extracted_str = json_match.group(0).strip()
        else:
            extracted_str = response_str.strip()
        
        # Try to parse JSON
        try:
            # First try direct parsing
            try:
                parsed_data = json.loads(extracted_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try json_repair if available
                try:
                    import json_repair
                    parsed_data = json_repair.loads(extracted_str)
                except ImportError:
                    # If json_repair is not available, try preprocessing
                    processed_str = self._preprocess_json_string(extracted_str)
                    parsed_data = json.loads(processed_str)
            
            if not isinstance(parsed_data, dict):
                raise ValueError(f"Parsed data is not a dictionary: {type(parsed_data)}")
            
            score = float(parsed_data.get('score', 0.0))
            reasoning = parsed_data.get('reasoning', "No reasoning provided")
            
            return (score, reasoning)
            
        except Exception as e:
            return (0.0, f"Parsing error: {e}")
        
    def _preprocess_json_string(self, json_str: str) -> str:
        """
        Preprocess JSON string to fix common formatting issues
        
        Args:
            json_str: JSON string to process
            
        Returns:
            str: Processed JSON string
        """
        # 1. Try to extract outermost JSON object (if any)
        json_obj_match = re.search(r'(\{.*\})', json_str, re.DOTALL)
        if json_obj_match:
            json_str = json_obj_match.group(1)
        
        # 2. Handle invalid escape sequences
        valid_escapes = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
        processed = ''
        i = 0
        while i < len(json_str):
            if json_str[i] == '\\' and i+1 < len(json_str):
                if json_str[i+1] in valid_escapes:
                    processed += json_str[i:i+2]
                    i += 2
                else:
                    processed += '\\\\' + json_str[i+1]
                    i += 2
            else:
                processed += json_str[i]
                i += 1
        
        # 3. Handle potential single quotes (replace with double quotes)
        processed = re.sub(r"(?<!\\)'([^']*)':", r'"\1":', processed)
        processed = re.sub(r':(?:\s*)\'([^\']*)\'', r':"\1"', processed)
        
        return processed
    
    def _evaluate_state_llm(self, problem_description: str, current_path: List[str]) -> Tuple[float, str]:
        """Internal helper function that calls LLM-based state evaluator and parses results.
        Returns a tuple (score, reasoning)"""
        if not self.evaluate_state:
            raise ValueError("LLM-based state evaluation requires self.evaluate_state to be configured.")

        current_path_str = "\n-> ".join(current_path) if current_path else "Initial state."
        try:
            # Use ChatAdapter during this call
            with dspy.context(adapter=dspy.ChatAdapter()):
                prediction = self.evaluate_state(
                    problem_description=problem_description,
                    current_path=current_path_str
                )

            # Access the 'evaluation_results' output field defined in the signature
            evaluation_results_str = prediction.evaluation_results
            
            # Use the new method to extract and parse JSON
            return self._extract_and_parse_json(evaluation_results_str)
            
        except Exception as e:
            return (0.0, f"Evaluation error: {e}")
    
    async def _evaluate_state_llm_async(self, problem_description: str, current_path: List[str]) -> Tuple[float, str]:
        """Async version of the evaluation function for parallel evaluation"""
        # Run synchronous evaluation function in ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                self._evaluate_state_llm, 
                problem_description, 
                current_path
            )
    
    async def _parallel_evaluate_paths(self, problem_description: str, paths_to_evaluate: List[List[str]]) -> List[Tuple[List[str], float, str]]:
        """Evaluate multiple paths in parallel, return list of (path, score, reasoning) tuples"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def evaluate_with_semaphore(path):
            async with semaphore:
                score, reasoning = await self._evaluate_state_llm_async(problem_description, path)
                return (path, score, reasoning)
        
        tasks = [evaluate_with_semaphore(path) for path in paths_to_evaluate]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                continue
            valid_results.append(result)
        
        return valid_results
    
    def _execute_parallel_eval(self, problem_description: str, paths_to_evaluate: List[List[str]]) -> List[Tuple[List[str], float, str]]:
        """
        Safely execute parallel evaluation, handling event loop issues
        """
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return [(path, *self._evaluate_state_llm(problem_description, path)) for path in paths_to_evaluate]
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(self._parallel_evaluate_paths(problem_description, paths_to_evaluate))
            
            if loop != asyncio.get_event_loop():
                loop.close()
                
            return results
            
        except Exception:
            return [(path, *self._evaluate_state_llm(problem_description, path)) for path in paths_to_evaluate]
    
    def _parse_thoughts(self, thoughts_output: Any) -> List[str]:
        """Parse output from thought generation into a list of distinct thoughts"""
        if isinstance(thoughts_output, list):
            stringified_thoughts = [str(item) for item in thoughts_output]
            return stringified_thoughts[:self.num_branches]
        elif isinstance(thoughts_output, str):
            lines = thoughts_output.strip().split('\n')
            parsed_thoughts = []
            for line in lines:
                cleaned_line = re.sub(r"^\s*[\*\-\–\d#]+\.?\s*", "", line).strip()
                if cleaned_line:
                    parsed_thoughts.append(cleaned_line)
            if len(parsed_thoughts) >= 1:
                if len(parsed_thoughts) == 1 and '\n' in thoughts_output and len(lines) > self.num_branches:
                    return [thoughts_output.strip()][:self.num_branches]
                elif len(parsed_thoughts) > 0:
                    return parsed_thoughts[:self.num_branches]
            
            sentences = re.split(r'(?<=[.!?])\s+', thoughts_output.strip())
            non_empty_sentences = [s.strip() for s in sentences if s and s.strip()]
            if len(non_empty_sentences) > 1 and len(non_empty_sentences) <= self.num_branches * 2:
                return non_empty_sentences[:self.num_branches]
            else:
                return [thoughts_output.strip()]
        else:
            return []
    
    def _format_problem_description(self, **kwargs) -> str:
        """Format input kwargs into a problem description string"""
        return "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in kwargs.items()])
    
    async def aforward(self, **kwargs):
        """Async version of forward method"""
        raise NotImplementedError("Async execution not yet implemented for TreeOfThought")
            
    def forward(self, **kwargs) -> dspy.Prediction:
        """Execute tree of thought search to solve the problem"""
        # Reset tree and state
        self.root_node = TreeNode(thought="", depth=0)
        self.best_paths = []
        
        # Prepare input
        problem_description = self._format_problem_description(**kwargs)
        initial_state: List[str] = []

        BeamItem = Tuple[List[str], float]
        beam: List[BeamItem] = [(initial_state, 0.0)]
        completed_paths: List[BeamItem] = []
        
        # Start timing
        start_time = time.time()

        for depth in range(self.max_depth):
            candidates: List[BeamItem] = []
            beam_copy = list(beam)
            beam = []

            for state_path, current_score in beam_copy:
                current_path_str = "\n-> ".join(state_path) if state_path else "Initial state."

                # Find or create tree node for current path
                current_node = self.root_node
                for thought in state_path:
                    found = False
                    for child in current_node.children:
                        if child.thought == thought:
                            current_node = child
                            found = True
                            break
                    
                    if not found:
                        current_node = current_node.add_child(thought, current_score)

                # 1. Generate next thoughts
                try:
                    generated = self.generate_thoughts(
                        problem_description=problem_description,
                        current_path=current_path_str
                    )
                    next_thoughts_output = generated.next_thoughts
                    next_thoughts_list = self._parse_thoughts(next_thoughts_output)
                except Exception:
                    continue

                if not next_thoughts_list:
                    continue

                # 2. Prepare new paths to evaluate
                paths_to_evaluate = []
                for thought in next_thoughts_list:
                    new_path = state_path + [thought.strip()]
                    paths_to_evaluate.append(new_path)

                # 3. Evaluate all new paths in parallel or sequentially
                path_scores = []
                
                if self.parallel and len(paths_to_evaluate) > 1:
                    path_scores = self._execute_parallel_eval(problem_description, paths_to_evaluate)
                else:
                    for new_path in paths_to_evaluate:
                        score, reasoning = self._evaluate_state_llm(problem_description, new_path)
                        path_scores.append((new_path, score, reasoning))

                # 4. Process evaluation results
                for new_path, score, reasoning in path_scores:
                    # Create new node in tree
                    thought = new_path[-1]
                    current_node.add_child(thought, score, reasoning)
                    
                    # Check if score is valid
                    if not (score == score):  # NaN check
                        continue
                        
                    # Add to candidates list
                    is_complete = depth == self.max_depth - 1
                    item = (new_path, score)

                    if is_complete:
                        completed_paths.append(item)
                    
                    candidates.append(item)

            # 5. Beam pruning
            candidates.sort(key=lambda item: item[1], reverse=True)
            beam = candidates[:self.beam_width]
            
            if not beam:
                break

        # 6. Select final best path
        final_candidates = completed_paths + beam
        if not final_candidates:
            try:
                final_result_prediction = self.generate_final_answer(
                    problem_description=problem_description,
                    best_path="No available reasoning paths."
                )
                output_fields = {field: getattr(final_result_prediction, field, None) 
                                for field in self.signature.output_fields.keys()}
                return dspy.Prediction(**output_fields, best_path_trace="No available reasoning paths.", best_score=0.0)
            except Exception:
                empty_output = {field: None for field in self.signature.output_fields.keys()}
                return dspy.Prediction(**empty_output, best_path_trace="Failed", best_score=0.0)

        # Sort by score in descending order
        final_candidates.sort(key=lambda item: item[1], reverse=True)
        best_path, best_score = final_candidates[0]
        self.best_paths = [final_candidates[0]]
        
        best_path_str = "\n-> ".join(best_path) if best_path else "None"

        # 7. Generate final answer using best path
        try:
            final_result_prediction = self.generate_final_answer(
                problem_description=problem_description,
                best_path=best_path_str
            )
            output_fields = {field: getattr(final_result_prediction, field) 
                            for field in self.signature.output_fields.keys()}
        except Exception:
            failed_output = {field: f"Error: Failed to generate field '{field}'" 
                            for field in self.signature.output_fields.keys()}
            return dspy.Prediction(**failed_output, best_path_trace=best_path_str, best_score=best_score)

        # 8. Return final Prediction
        return dspy.Prediction(**output_fields, best_path_trace=best_path_str, best_score=best_score)


class TreeVisualizer:
    """Visualization utilities for Tree of Thought"""
    
    @staticmethod
    def visualize_tree_ascii(root_node, max_depth=None, highlight_best=True, best_paths=None):
        """
        Renders a tree visualization using ASCII characters
        
        Args:
            root_node: The root node of the tree
            max_depth: Maximum depth to visualize
            highlight_best: Whether to highlight the best path
            best_paths: List of best paths to highlight
        """
        if max_depth is None:
            # Find the maximum depth in the tree
            def find_max_depth(node):
                if not node.children:
                    return node.depth
                return max(find_max_depth(child) for child in node.children)
            max_depth = find_max_depth(root_node)
            
        print("\n" + "="*50)
        print("Tree of Thought Visualization (ASCII)")
        print("="*50)
        
        # Preprocess: prepare display info for each node
        def prepare_node_display(node, is_last_child=False):
            """Prepare display information for a node"""
            # Check if on best path
            is_best_path = False
            if highlight_best and best_paths:
                best_path_thoughts = best_paths[0][0]  # Get thoughts from best path
                node_path = node.path_to_root()
                is_best_path = all(i < len(best_path_thoughts) and p == best_path_thoughts[i] 
                                for i, p in enumerate(node_path) if p)
            
            # Build node display text
            if not node.thought:
                node_text = "Root"
            else:
                node_text = node.thought
            
            # Choose prefix symbol based on score
            if node.score >= 0.8:
                score_prefix = "✓ "  # High score
            elif node.score >= 0.5:
                score_prefix = "○ "  # Medium score
            elif node.score > 0:
                score_prefix = "△ "  # Low score
            else:
                score_prefix = "  "  # Unscored
            
            # Add special marker for best path
            if is_best_path:
                best_marker = "* "
            else:
                best_marker = "  "
            
            # Build complete node text
            full_text = f"{best_marker}{score_prefix}[{node.score:.2f}] {node_text}"
            
            return {
                "node": node,
                "text": full_text,
                "is_best": is_best_path,
                "is_last_child": is_last_child
            }
        
        def print_tree(node, prefix="", is_last=True, node_info=None):
            """Recursively print tree structure"""
            if node_info is None:
                node_info = prepare_node_display(node, is_last)
            
            # Determine connector for current line
            connector = "└── " if is_last else "├── "
            
            # Print current node
            if node.depth == 0:
                # Special handling for root node
                print(f"● Root")
                next_prefix = ""
            else:
                # Non-root node with connection lines
                if node_info["is_best"]:
                    print(f"{prefix}{connector}*{node_info['text'][1:]}")  # Add star for best path
                else:
                    print(f"{prefix}{connector}{node_info['text']}")
                
                # Determine prefix for next level
                next_prefix = prefix + ("    " if is_last else "│   ")
            
            # Process child nodes
            children = node.children
            for i, child in enumerate(children):
                child_is_last = i == len(children) - 1
                child_info = prepare_node_display(child, child_is_last)
                print_tree(child, next_prefix, child_is_last, child_info)
        
        # Start printing tree
        print_tree(root_node)
        print("\n" + "-"*50)
        
        # Print statistics and legend
        def count_nodes(node):
            """Count total nodes in tree"""
            return 1 + sum(count_nodes(child) for child in node.children)
            
        total_nodes = count_nodes(root_node)
        print(f"Tree stats: Total nodes: {total_nodes}, Max depth: {max_depth}")
        
        # Print legend
        print("\nLegend:")
        print("* - Best path node")
        print("✓ - High score node (≥0.8)")
        print("○ - Medium score node (≥0.5)")
        print("△ - Low score node (<0.5)")
        print("="*50)
        
        return total_nodes, max_depth


'''
# Basic usage
tot = TreeOfThought("question -> answer", max_depth=1, num_branches=1, beam_width=1)
result = tot(question="What is 1+1?")

print(result.answer)
TreeVisualizer.visualize_tree_ascii(tot.root_node, best_paths=tot.best_paths) # optional

'''