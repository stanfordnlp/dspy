import random
import re

import dspy
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.utils import create_example_string, create_predictor_level_history_string, strip_prefix
from dspy.teleprompt.utils import get_signature

from .propose_base import Proposer

# Hardcoded variables (TODO: update)
MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
        "none": "",
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }

### SIGNATURES USED TO HELP WITH INSTRUCTION GENERATION ###

class DescribeProgram(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.OutputField(
        desc="Describe what task the program is designed to solve, and how it goes about solving this task.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )


class DescribeModule(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.InputField(
        desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )
    module = dspy.InputField(
        desc="The module in the program that we want to describe.", prefix="MODULE:",
    )
    module_description = dspy.OutputField(
        desc="Description of the module's role in the broader program.",
        prefix="MODULE DESCRIPTION:",
    )


def generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    class GenerateSingleModuleInstruction(dspy.Signature):
        (
            """Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task."""
        )
        if use_dataset_summary:
            dataset_description = dspy.InputField(
                desc="A description of the dataset that we are using.",
                prefix="DATASET SUMMARY:",
            )
        if program_aware:
            program_code = dspy.InputField(
                format=str,
                desc="Language model program designed to solve a particular task.",
                prefix="PROGRAM CODE:",
            )
            program_description = dspy.InputField(
                desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
                prefix="PROGRAM DESCRIPTION:",
            )
            module = dspy.InputField(
                desc="The module to create an instruction for.", prefix="MODULE:",
            )
        task_demos = dspy.InputField(
            format=str,
            desc="Example inputs/outputs of our module.",
            prefix="TASK DEMO(S):",
        )
        if use_instruct_history:
            previous_instructions = dspy.InputField(
                format=str,
                desc="Previous instructions we've attempted, along with their associated scores.",
                prefix="PREVIOUS INSTRUCTIONS:",
            )
        basic_instruction = dspy.InputField(
            format=str, desc="Basic instruction.", prefix="BASIC INSTRUCTION:",
        )
        if use_tip:
            tip = dspy.InputField(
                format=str,
                desc="A suggestion for how to go about generating the new instruction.",
                prefix="TIP:",
            )
        proposed_instruction = dspy.OutputField(
            desc="Propose an instruction that will be used to prompt a Language Model to perform this task.",
            prefix="PROPOSED INSTRUCTION:",
        )

    return dspy.Predict(GenerateSingleModuleInstruction)

### CLASS RESPONSIBLE FOR GENERATING A NEW INSTRUCTION, USING THE HELPER SIGNATURES ABOVE ###

class GenerateModuleInstruction(dspy.Module):
    def __init__(
        self,
        program_code_string=None,
        use_dataset_summary=True,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
    ):
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip

        self.program_code_string = program_code_string
        self.describe_program = dspy.Predict(DescribeProgram)
        self.describe_module = dspy.Predict(DescribeModule)
        self.generate_module_instruction = generate_instruction_class(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )

    def forward(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        max_demos=3,
        tip=None,
    ):
        # Construct full program demo or single module demo depending on whether or not we're using the full program
        task_demos = ""
        basic_instruction = get_signature(program.predictors()[pred_i]).instructions
        curr_demos_num = 0
        
        for example in demo_candidates[pred_i][demo_set_i]:
            if "augmented" in example.keys():
                fields_to_use = get_signature(program.predictors()[pred_i]).fields
                example_string = create_example_string(fields_to_use, example)
                task_demos += f"{example_string}\n"
                curr_demos_num += 1
                if curr_demos_num >= max_demos:
                    break

        # Summarize the program
        program_description = ""
        module_code = ""
        if self.program_aware:
            program_description = strip_prefix(
                self.describe_program(
                    program_code=self.program_code_string, program_example=task_demos,
                ).program_description,
            )
            print(f"PROGRAM DESCRIPTION: {program_description}")

            # Identify all modules
            init_pattern = r"def __init__\(.*?\):([\s\S]*?)(?=^\s*def|\Z)"
            init_content_match = re.search(init_pattern, self.program_code_string)
            init_content = init_content_match.group(0)
            pattern = r"^(.*dspy\.(ChainOfThought|Predict).*)$"  # TODO: make it so that this extends out to any dspy Module
            matches = re.findall(pattern, init_content, re.MULTILINE)
            modules = [match[0].strip() for match in matches]
            module_code = modules[pred_i]

        module_description = self.describe_module(
            program_code=self.program_code_string,
            program_description=program_description,
            program_example=task_demos,
            module=module_code,
            max_depth=10,
        ).module_description

        # Generate an instruction for our chosen module
        print(f"task_demos {task_demos}")
        instruct = self.generate_module_instruction(
            dataset_description=data_summary,
            program_code=self.program_code_string,
            program_description=program_description,
            module=module_code,
            task_demos=task_demos,
            tip=tip,
            basic_instruction=basic_instruction,
            previous_instructions=previous_instructions,
            module_description=module_description,
        )
        if hasattr(instruct, "module_description"):
            module_description = strip_prefix(instruct.module_description)
            print(f"MODULE DESCRIPTION: {module_description}")
        proposed_instruction = strip_prefix(instruct.proposed_instruction)
        # print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return dspy.Prediction(proposed_instruction=proposed_instruction)

### CLASS USED TO GENERATE THE FULL SET OF INSTRUCTIONS GIVEN THE SPECIFIED CRITERIA ###

class GroundedProposer(Proposer):
    def __init__(
        self,
        prompt_model,
        trainset,
        program_code_string=None,
        view_data_batch_size=10,
        use_dataset_summary=True,
        program_aware=True,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
    ):
        super().__init__()
        self.program_aware = program_aware
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly=set_tip_randomly
        self.set_history_randomly=set_history_randomly

        self.prompt_model = prompt_model
        self.program_code_string = program_code_string
        self.data_summary = create_dataset_summary(
            trainset=trainset, view_data_batch_size=view_data_batch_size, prompt_model=prompt_model,
        )
        print(f"DATA SUMMARY: {self.data_summary}")

    def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        prompt_model,
        trial_logs,
        N,
        T,
        tip=None,
    ):
        """This method is responsible for returning the full set of new instructions for our program, given the specified criteria."""

        proposed_instructions = {}

        if self.set_tip_randomly:
            print("Using a randomly generated configuration for our grounded proposer.")
            # Randomly select the tip
            selected_tip_key = random.choice(list(TIPS.keys()))
            selected_tip = TIPS[selected_tip_key]
            self.use_tip = bool(
                selected_tip,
            )
            print(f"Selected tip: {selected_tip_key}")        

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = random.random() < 0.5
            self.use_instruct_history = use_history
            print(f"Use history T/F: {self.use_instruct_history}")

        # Create an instruction for each predictor 
        for pred_i, predictor in enumerate(program.predictors()):
            for demo_set_i in range(len(demo_candidates[0])):
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                proposed_instructions[pred_i].append(
                    self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        prompt_model=prompt_model,
                        T=T,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                    ),
                )
        return proposed_instructions

    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        prompt_model,
        T,
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
    ):
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        # Create an instruction history string for our predictor
        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY,
        )

        # Create our instruction generator class (given specific criteria for this round of proposal)
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
        )

        # Generate a new instruction for our predictor, using the temperature specified for this round
        original_temp = prompt_model.kwargs["temperature"]
        with dspy.settings.context(lm=prompt_model):
            prompt_model.kwargs["temperature"] = T
            proposed_instruction = instruction_generator.forward(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                tip=tip,
            ).proposed_instruction
        prompt_model.kwargs["temperature"] = original_temp

        # Log the trace used to generate the new instruction, along with the new instruction itself
        prompt_model.inspect_history(n=1)
        print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return strip_prefix(proposed_instruction)
