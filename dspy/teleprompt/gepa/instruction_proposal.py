
import dspy

class GenerateKBQueries(dspy.Signature):
    """Generate targeted knowledge base queries to improve model performance on a specific task. 
    
For the specific task, you are provided with a set of example inputs and the model's incorrect outputs, along with feedback explaining the issues with the outputs. Your task is to create targetted queries that retrieve information from a knowledge base to address specific feedback on model outputs. The goal is to find as much information from the knowledge base as possible to help the model improve its performance on the task."""
    task_samples: str = dspy.InputField()
    kb_queries: list[str] = dspy.OutputField()

class FetchFromKB(dspy.Module):
    def __init__(self, query_engine):
        # self.knowledge_index = knowledge_index
        self.query_engine = query_engine # knowledge_index.as_query_engine()
        # self.retriever = knowledge_index.as_retriever()
        self.gen_kb_queries = dspy.ChainOfThought(GenerateKBQueries)
    
    def forward(self, samples_to_search_for):
        o = self.gen_kb_queries(task_samples=samples_to_search_for)
        queries = o.kb_queries
        queries = queries[:min(5, len(queries))]
        responses = []
        for query in queries:
            query = "Answer the following query in detail: " + query + "\n\nInclude as much relevant information from the documents as possible."
            response = self.query_engine.query(query).response
            responses.append(response)
        
        return dspy.Prediction(queries=queries, responses=responses)


prompt1 = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

prompt2 = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Some additional information about the task from reference materials:
```
<reference_materials>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

# class ImproveInstruction(dspy.Signature):
#     """Your task is to **create a comprehensive, clear, and fully self-contained instruction document** that enables people with no prior experience to successfully perform a complex task. The instructions must ensure that users can complete the task accurately using only the document you produce without requiring any external sources.
# ---
# ## What You Will Receive
# You will be provided with:
# - **Original Instructions:**
# The previous version of the instruction document.
# - **User Examples and Feedback:**
# Outputs from users of the previous instruction document, with notes highlighting common mistakes and misunderstandings stemming from incompleteness of the previous guidance document as feedback.
# - **Reference Materials:**
# Any other information relevant to the task for improving the document.
# ---
# ## What You Must Produce
# Develop a **detailed instruction document** that:
# - **Is self-contained.**
# All needed definitions, rules, and context are included. Do not refer to any external documents or sources.
# - **Addresses all prior confusion.**
# Integrate explicit corrections based on user feedback and examples of past errors.
# - **Uses clear, precise language.**
# - **Includes practical examples.**
# Show both correct and incorrect task execution, with explanations.
# - **Explains all key terms and concepts.**
# - **States all requirements.**
# Clearly specify any output formatting requirements, acceptable outputs, and evaluation criteria.
# - **Covers edge cases and exceptions.**
# - **Includes all required domain knowledge from feedbacks and reference material**
# - **Includes repeatable task patterns in how previous users attempted the task correctly**
# - **Do not waste space in formatting and headers of the document. Be direct.**
# Follow these directions to produce instructions that enable any user—without any prior knowledge of task domain—to complete the task successfully and consistently."""
#     current_instruction_doc = dspy.InputField(
#         desc="The current (previous) version of the instruction document that requires improvement.",
#         format=str
#     )
#     user_examples_and_feedback = dspy.InputField(
#         desc="User-generated example outputs and corresponding feedback, indicating recurring mistakes, misunderstandings, and any confusion about previous instructions.",
#         format=str
#     )
#     improved_instruction_doc = dspy.OutputField(
#         desc="A comprehensive, self-contained, and clear instruction document that resolves all identified user issues, defines all necessary concepts, provides explicit examples (correct and incorrect with explanations), specifies all requirements, and incorporates all provided domain and feedback-informed guidance.",
#         format=str
#     )

def call_lm_and_extract_response(prompt, lm, current_instruction_doc, user_examples_and_feedback, reference_materials=None):
    full_prompt = prompt.replace("<curr_instructions>", current_instruction_doc)
    full_prompt = full_prompt.replace("<inputs_outputs_feedback>", user_examples_and_feedback)
    if reference_materials is not None:
        full_prompt = full_prompt.replace("<reference_materials>", reference_materials)
    
    lm_out = lm(full_prompt)[0]
    # Extract ``` blocks
    start = lm_out.find("```")
    end = lm_out.rfind("```")
    if start == -1 or end == -1:
        return lm_out
    else:
        return lm_out[start+3:end].strip()

class ProposeNewInstructionModule:
    def __init__(self, base_program, instruction_lm, dataset_with_feedback, knowledgebase_qe, kb_fetch_lm=None):
        self.base_program = base_program
        self.dataset_with_feedback = dataset_with_feedback

        signature_to_use = prompt1 # ImproveInstruction
        if knowledgebase_qe is not None:
            # signature_to_use = signature_to_use.append(
            #     "reference_materials",
            #     dspy.InputField(
            #         desc="Additional domain knowledge, context, specifications, or best practices that should inform the improved instructions.",
            #         format=str
            #     )
            # )
            signature_to_use = prompt2
            self.kb_query = knowledgebase_qe
            self.kb_fetch = FetchFromKB(knowledgebase_qe)
            if kb_fetch_lm is not None:
                self.kb_fetch.set_lm(kb_fetch_lm)
        else:
            self.kb_query = None
            self.kb_fetch = None

        # self.instruction_propose_module = dspy.Predict(signature_to_use)
        self.instruction_lm = instruction_lm
    
    def format_samples(self, samples):
        def convert_sample_to_markdown(sample, examplenum):
            s = "# Example " + str(examplenum) + "\n"
            s += "## Inputs\n"
            contains_history = False
            history_key_name = None
            for input_key, input_val in sample['inputs'].items():
                if isinstance(input_val, dspy.History):
                    contains_history = True
                    assert history_key_name is None
                    history_key_name = input_key
            
            if contains_history:
                s += f"### Context\n"
                s += "```json\n"
                for i, message in enumerate(sample['inputs'][history_key_name].messages):
                    s += f"  {i}: {message}\n"
                s += "```\n\n"
            
            for input_key, input_val in sample['inputs'].items():
                if contains_history and input_key == history_key_name:
                    continue
                s += f"### {input_key}\n"
                s += str(input_val) + "\n\n"
            
            s += "## Generated Outputs\n"
            for output_key, output_val in sample['generated_output'].items():
                s += f"### {output_key}\n"
                s += str(output_val) + "\n\n"
            
            s += "## Feedback\n"
            s += sample['feedback'] + "\n\n"

            return s
        
        return "\n\n".join([convert_sample_to_markdown(sample, i+1) for i, sample in enumerate(samples)])

    def fetch_relevant_information(self, samples):
        qe_response = self.kb_fetch(samples_to_search_for=self.format_samples(samples))
        response = "\n".join(qe_response.responses)
        return response
    
    def compile(self):
        instruction = self.base_program.signature.instructions
        
        sample = self.dataset_with_feedback
        if self.kb_query:
            kb_info = self.fetch_relevant_information(sample)
            # module_output = self.instruction_propose_module(
            #     current_instruction_doc=instruction,
            #     user_examples_and_feedback=self.format_samples(sample),
            #     reference_materials=kb_info
            # )
            new_instruction = call_lm_and_extract_response(
                prompt2,
                self.instruction_lm,
                current_instruction_doc=instruction,
                user_examples_and_feedback=self.format_samples(sample),
                reference_materials=kb_info
            )
        else:
            kb_info = "No QueryEngine Provided"
            # module_output = self.instruction_propose_module(
            #     current_instruction_doc=instruction,
            #     user_examples_and_feedback=self.format_samples(sample),
            # )
            new_instruction = call_lm_and_extract_response(
                prompt1,
                self.instruction_lm,
                current_instruction_doc=instruction,
                user_examples_and_feedback=self.format_samples(sample)
            )
            
        # new_instruction = module_output.improved_instruction_doc

        return {
            "new_instruction": new_instruction,
            "module_output": {"improved_instruction_doc": new_instruction},
            "kb_info": kb_info,
            "all_inputs": {
                "current_instruction_doc": instruction,
                "user_examples_and_feedback": self.format_samples(sample),
                "reference_materials": kb_info
            }
        }
