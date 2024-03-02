import dsp
import dspy
from dspy.signatures.signature import signature_to_template
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.signatures import Signature
from dspy.evaluate.evaluate import Evaluate
from collections import defaultdict
import random
from dspy.teleprompt import BootstrapFewShot
import optuna
import math

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter using the BayesianSignatureOptimizer, and evaluate it on an end task:

from dspy.teleprompt import BayesianSignatureOptimizer

teleprompter = BayesianSignatureOptimizer(prompt_model=prompt_model, task_model=task_model, metric=metric, n=10, init_temperature=1.0)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program, devset=devset[:DEV_NUM], optuna_trials_num=100, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* task_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* n: The number of new prompts and sets of fewshot examples to generate and evaluate. Default=10.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.0.
* verbose: Tells the method whether or not to print intermediate steps.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track a dictionary with a key corresponding to the trial number, 
                and a value containing a dict with the following keys:
                    * program: the program being evaluated at a given trial
                    * score: the last average evaluated score for the program
                    * pruned: whether or not this program was pruned
                This information will be returned as attributes of the best program.
"""
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithDataObservations(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English.  I will also give you some ``observations`` I have made about the dataset and task. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    observations = dspy.InputField(desc="Observations about the dataset and task")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithExamples(dspy.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will also provide you with the current ``basic instruction`` that is being used for this task. I will also provide you with some ``examples`` of the expected inputs and outputs.

Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        # attempted_instructions = dspy.InputField(format=str, desc="Previously attempted task instructions, along with their resulting validation score, and an example of the instruction in use on a sample from our dataset.")
        basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
        # examples = dspy.InputField(format=dsp.passages2text, desc="Example(s) of the task")
        examples = dspy.InputField(format=dsp.passages2text, desc="Example(s) of the task")
        proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class BasicGenerateInstructionWithExamplesAndDataObservations(dspy.Signature):
        ("""You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Specifically, I will also provide you with the current ``basic instruction`` that is being used for this task. I will also provide you with some ``observations`` I have made about the dataset and task, along with some ``examples`` of the expected inputs and outputs.

Your task is to propose a new improved instruction and prefix for the output field that will lead a good language model to perform the task well. Don't be afraid to be creative.""")
        basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
        observations = dspy.InputField(desc="Observations about the dataset and task")
        examples = dspy.InputField(format=dsp.passages2text, desc="Example(s) of the task")
        proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class ObservationSummarizer(dspy.Signature):
    ("""Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.""")
    observations = dspy.InputField(desc="Observations I have made about my dataset")
    summary = dspy.OutputField(desc="Two to Three sentence summary of only the most significant highlights of my observations")

class DatasetDescriptor(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")
    
    examples = dspy.InputField(desc="Sample data points from the dataset")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed")

class DatasetDescriptorWithPriorObservations(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """I will also provide you with a few observations I have already made.  Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE' """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")
    
    examples = dspy.InputField(desc="Sample data points from the dataset")
    prior_observations = dspy.InputField(desc="Some prior observations I made about the data")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add")

class BayesianSignatureOptimizer(Teleprompter):
    def __init__(self, prompt_model=None, task_model=None, teacher_settings={}, n=10, metric=None, init_temperature=1.0, verbose=False, track_stats=False, view_data_batch_size=10):
        self.n = n
        self.metric = metric
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model if prompt_model is not None else dspy.settings.lm
        self.task_model = task_model if task_model is not None else dspy.settings.lm
        self.verbose = verbose
        self.track_stats = track_stats
        self.teacher_settings = teacher_settings
        self.view_data_batch_size = view_data_batch_size
        
    def _print_full_program(self, program):
        for i,predictor in enumerate(program.predictors()):
            if self.verbose: print(f"Predictor {i}")
            if (hasattr(predictor, 'extended_signature')):
                if self.verbose: print(f"i: {predictor.extended_signature.instructions}")
                *_, last_field = predictor.extended_signature.fields.values()
                if self.verbose: print(f"p: {last_field.json_schema_extra['prefix']}")
            else:
                if self.verbose: print(f"i: {predictor.extended_signature1.instructions}")
                *_, last_field = predictor.extended_signature1.fields.values()
                if self.verbose: print(f"p: {last_field.json_schema_extra['prefix']}")
            if self.verbose: print("\n")
    
    def _print_model_history(self, model, n=1):
        if self.verbose: print(f"Model ({model}) History:")
        model.inspect_history(n=n)

    def _observe_data(self, trainset):
        upper_lim = min(len(trainset), self.view_data_batch_size)
        observation = dspy.Predict(DatasetDescriptor, n=1, temperature=1.0)(examples=(trainset[0:upper_lim].__repr__()))
        observations = observation["observations"]

        skips = 0
        for b in range(self.view_data_batch_size, len(trainset), self.view_data_batch_size):
            upper_lim = min(len(trainset), b+self.view_data_batch_size)
            output = dspy.Predict(DatasetDescriptorWithPriorObservations, n=1, temperature=1.0)(prior_observations=observations, examples=(trainset[b:upper_lim].__repr__()))
            if len(output["observations"]) >= 8 and output["observations"][:8].upper() == "COMPLETE":
                skips += 1
                if skips >= 5:
                    break
                continue
            observations += output["observations"]

        summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(observations=observations)

        return summary.summary
    
    def _create_example_string(self, fields, example):

        # Building the output string
        output = []
        for field in fields:
            name = field.name
            separator = field.separator
            input_variable = field.input_variable

            # Determine the value from input_data or prediction_data
            value = example.get(input_variable)

            # Construct the string for the current field
            field_str = f"{name}{separator}{value}"
            output.append(field_str)

        # Joining all the field strings
        return '\n'.join(output)
    
    def _generate_first_N_candidates(self, module, N, view_data, view_examples, demo_candidates, devset):
        candidates = {}
        evaluated_candidates = defaultdict(dict)

        if view_data:
            # Create data observations
            self.observations = None
            with dspy.settings.context(lm=self.prompt_model):
                self.observations = self._observe_data(devset).replace("Observations:","").replace("Summary:","")
            
        if view_examples:
            example_sets = {}
            for predictor in module.predictors():
                # Get all augmented examples
                example_set = {}
                all_sets_of_examples = demo_candidates[id(predictor)] # Get all generated sets of examples
                for example_set_i, set_of_examples in enumerate(all_sets_of_examples):
                    if example_set_i != 0: # Skip the no examples case
                        for example in set_of_examples: # Get each individual example in the set
                            if "augmented" in example.keys():
                                if example["augmented"]:
                                    if example_set_i not in example_set:
                                        example_set[example_set_i] = []
                                    fields_to_use = signature_to_template(predictor.signature).fields
                                    input_variable_names = list(predictor.signature.input_fields.keys())
                                    example_with_only_signature_fields = {key: value for key, value in example.items() if key in input_variable_names}
                                    example_string = self._create_example_string(fields_to_use, example_with_only_signature_fields)
                                    example_set[example_set_i].append(example_string)
                        example_sets[id(predictor)] = example_set  
                    else:
                        example_set[example_set_i] = []
                        example_sets[id(predictor)] = example_set

        # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            if (hasattr(predictor, 'extended_signature')):
                basic_instruction = predictor.extended_signature.instructions
                *_, last_field = predictor.extended_signature.fields.values()
                basic_prefix = last_field.json_schema_extra["prefix"]
            else:
                basic_instruction = predictor.extended_signature1.instructions
                *_, last_field = predictor.extended_signature1.fields.values()
                basic_prefix = last_field.json_schema_extra["prefix"]
            with dspy.settings.context(lm=self.prompt_model):
                # Data & Examples
                if view_data and view_examples:
                    if 1 not in example_sets[id(predictor)].keys():
                        raise ValueError("No examples found for the given predictor")
                    instruct = None
                    for i in range(1, self.n):
                        new_instruct = dspy.Predict(
                            BasicGenerateInstructionWithExamplesAndDataObservations,
                            n=1,
                            temperature=self.init_temperature,
                        )(
                            basic_instruction=basic_instruction,
                            observations=self.observations,
                            examples=example_sets[id(predictor)][i],
                        )
                        if not instruct:
                            instruct = new_instruct
                        else:
                            instruct.completions.proposed_instruction.extend(new_instruct.completions.proposed_instruction)
                            instruct.completions.proposed_prefix_for_output_field.extend(new_instruct.completions.proposed_prefix_for_output_field)
                # Just data
                elif view_data: 
                    instruct = dspy.Predict(BasicGenerateInstructionWithDataObservations, n=N-1, temperature=self.init_temperature)(basic_instruction=basic_instruction, observations=self.observations)
                # Just examples
                elif view_examples: 
                    instruct = None
                    for i in range(1,self.n): # Note: skip over the first example set which is empty
                        new_instruct = dspy.Predict(
                            BasicGenerateInstructionWithExamples,
                            n=1,
                            temperature=self.init_temperature,
                        )(
                            basic_instruction=basic_instruction,
                            examples=example_sets[id(predictor)][i],
                        )
                        if not instruct:
                            instruct = new_instruct
                        else:
                            instruct.completions.proposed_instruction.extend(new_instruct.completions.proposed_instruction)
                            instruct.completions.proposed_prefix_for_output_field.extend(new_instruct.completions.proposed_prefix_for_output_field)
                # Neither
                else: 
                    instruct = dspy.Predict(BasicGenerateInstruction, n=N-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            
            # Add in our initial prompt as a candidate as well
            instruct.completions.proposed_instruction.insert(0, basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.insert(0, basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}
        
        if self.verbose: self._print_model_history(self.prompt_model)
        
        return candidates, evaluated_candidates

    def compile(self, student, *, devset, optuna_trials_num, max_bootstrapped_demos, max_labeled_demos, eval_kwargs, seed=42, view_data=True, view_examples=True):

        random.seed(seed)
        
        # Set up program and evaluation function
        module = student.deepcopy()
        evaluate = Evaluate(devset=devset, metric=self.metric, **eval_kwargs)

        # Generate N few shot example sets
        demo_candidates = {}
        for i in range(self.n):
            if i == 0: # Story empty set of demos as default for index 0
                for module_p in module.predictors():
                    if id(module_p) not in demo_candidates:
                        demo_candidates[id(module_p)] = []
                    demo_candidates[id(module_p)].append([])
            else:
                if self.verbose: print(f"Creating basic bootstrap: {i}/{self.n-1}")

                # Create a new basic bootstrap few - shot program .
                rng = random.Random(i)
                shuffled_devset = devset[:]  # Create a copy of devset
                rng.shuffle(shuffled_devset)  # Shuffle the copy
                tp = BootstrapFewShot(metric = self.metric, max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos, teacher_settings=self.teacher_settings)
                candidate_program = tp.compile(student=module.deepcopy(), trainset=shuffled_devset)

                # Store the candidate demos
                for module_p, candidate_p in zip(module.predictors(), candidate_program.predictors()):
                    if id(module_p) not in demo_candidates:
                        demo_candidates[id(module_p)] = []
                    demo_candidates[id(module_p)].append(candidate_p.demos)

        # Generate N candidate prompts
        instruction_candidates, _ = self._generate_first_N_candidates(module, self.n, view_data, view_examples, demo_candidates, devset)

        # Initialize variables to store the best program and its score
        best_score = float('-inf')
        best_program = None
        trial_num = 0

        trial_logs = {}

        # Define our trial objective
        def create_objective(baseline_program, instruction_candidates, demo_candidates, evaluate, devset):
            def objective(trial):
                nonlocal best_program, best_score, trial_num, trial_logs  # Allow access to the outer variables
                candidate_program = baseline_program.deepcopy()

                # Suggest the instruction to use for our predictor 
                if self.verbose: print(f"Starting trial num: {trial_num}")
                trial_logs[trial_num] = {}

                for p_old, p_new in zip(baseline_program.predictors(), candidate_program.predictors()):

                    # Get instruction candidates for our given predictor
                    p_instruction_candidates = instruction_candidates[id(p_old)]
                    p_demo_candidates = demo_candidates[id(p_old)]

                    # Suggest the index of the instruction candidate to use in our trial
                    #instruction_idx = trial.suggest_categorical(f"{id(p_old)}_predictor_instruction",range(len(p_instruction_candidates)))
                    #demos_idx = trial.suggest_categorical(f"{id(p_old)}_predictor_demos",range(len(p_demo_candidates)))
                    instruction_idx = trial.suggest_int(f"{id(p_old)}_predictor_instruction",low=0, high=len(p_instruction_candidates)-1)
                    demos_idx = trial.suggest_int(f"{id(p_old)}_predictor_demos",low=0, high=len(p_demo_candidates)-1)

                    trial_logs[trial_num][f"{id(p_old)}_predictor_instruction"] = instruction_idx
                    trial_logs[trial_num][f"{id(p_old)}_predictor_demos"] = demos_idx

                    # Get the selected instruction candidate 
                    selected_candidate = p_instruction_candidates[instruction_idx]
                    selected_instruction = selected_candidate.proposed_instruction.strip('"').strip()
                    selected_prefix = selected_candidate.proposed_prefix_for_output_field.strip('"').strip()

                    # Use this candidates in our program
                    *_, last_field = p_new.extended_signature.fields.keys()
                    p_new.extended_signature = p_new.extended_signature \
                        .with_instructions(selected_instruction) \
                        .with_updated_fields(last_field, prefix=selected_prefix)

                    # Get the selected demos
                    selected_demos = p_demo_candidates[demos_idx]

                    # Use these demos in our program
                    p_new.demos = selected_demos
                
                if self.verbose: print("Evaling the following program:")
                self._print_full_program(candidate_program)
                trial_logs[trial_num]["program"] = candidate_program

                # Evaluate with the new prompts
                total_score = 0
                batch_size = 100
                num_batches = math.ceil(len(devset) / batch_size)

                for i in range(num_batches):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, len(devset))
                    split_dev = devset[start_index:end_index]
                    split_score = evaluate(candidate_program, devset=split_dev, display_table=0)
                    if self.verbose: print(f"{i}st split score: {split_score}")

                    total_score += split_score * len(split_dev)
                    curr_weighted_avg_score = total_score / min((i+1)*100,len(devset))
                    if self.verbose: print(f"curr average score: {curr_weighted_avg_score}")

                    trial.report(curr_weighted_avg_score, i)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        if self.verbose: print("Optuna decided to prune!")
                        trial_logs[trial_num]["score"] = curr_weighted_avg_score
                        trial_logs[trial_num]["pruned"] = True
                        trial_num += 1 
                        raise optuna.TrialPruned()
                
                if self.verbose:
                    print(f"Fully evaled score: {curr_weighted_avg_score}")
                    self._print_model_history(self.task_model, n=1)
                score = curr_weighted_avg_score
                
                trial_logs[trial_num]["score"] = curr_weighted_avg_score
                trial_logs[trial_num]["pruned"] = False
            
                # Update the best program if the current score is better
                if score > best_score:
                    best_score = score
                    best_program = candidate_program.deepcopy()
                
                trial_num += 1 

                return score

            return objective

        # Run the trial 
        objective_function = create_objective(module, instruction_candidates, demo_candidates, evaluate, devset)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        score = study.optimize(objective_function, n_trials=optuna_trials_num)

        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs

        return best_program
