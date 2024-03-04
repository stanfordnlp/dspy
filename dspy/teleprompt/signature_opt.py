from collections import defaultdict

import dsp
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = SignatureOptimizer(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), devset=devset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to generate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* verbose: Tells the method whether or not to print intermediate steps.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(dspy.Signature):
        """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

        attempted_instructions = dspy.InputField(format=dsp.passages2text)
        proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class SignatureOptimizer(Teleprompter):
    def __init__(self, prompt_model=None, metric=None, breadth=10, depth=3, init_temperature=1.4, verbose=False, track_stats=False):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.verbose = verbose
        self.track_stats = track_stats

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors()):
            if p1.extended_signature.instructions != p2.extended_signature.instructions:
                return False
            *_, p1_last_field = p1.extended_signature.fields.values()
            *_, p2_last_field = p2.extended_signature.fields.values()
            if p1_last_field != p2_last_field:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c['score'] == last_batch_score:
                for c2 in last_batch:
                    if (self._check_candidates_equal(c, c2)):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c['score']
            if not repeat:
                final_candidates.append(c)
        return final_candidates

    def _print_signature(self, predictor):
        if self.verbose:
            if (hasattr(predictor, 'extended_signature')):
                signature = predictor.extended_signature
            else:
                signature = predictor.extended_signature1
            print(f"i: {signature.instructions}")
            print(f"p: {list(signature.fields.values())[-1].json_schema_extra['prefix']}")
            print()

    
    def compile(self, student, *, devset, eval_kwargs):
        """student is a program that needs to be optimized, note that it may be zero-shot or already pre-optimized for demos != []"""
        module = student.deepcopy()
        evaluate = Evaluate(devset=devset, metric=self.metric, **eval_kwargs)
        total_calls = 0
        results_best = {id(p):{"depth": [], "max": [], "average": [], "min":[], "std": []} for p in module.predictors()}
        results_latest = {id(p):{"depth": [], "max": [], "average": [], "min":[], "std": []} for p in module.predictors()}

        if self.track_stats:
            import numpy as np


        candidates = {}
        evaluated_candidates = defaultdict(dict)

        # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            *_, last_key = predictor.extended_signature.fields.keys()
            if (hasattr(predictor, 'extended_signature')):
                basic_instruction = predictor.extended_signature.instructions
                basic_prefix = predictor.extended_signature.fields[last_key].json_schema_extra['prefix']
            else:
                basic_instruction = predictor.extended_signature1.instructions
                basic_prefix = predictor.extended_signature1.fields[last_key].json_schema_extra['prefix']
            if self.prompt_model: 
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(BasicGenerateInstruction, n=self.breadth-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(BasicGenerateInstruction, n=self.breadth-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            # Add in our initial prompt as a candidate as well
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}
        
        if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")

        latest_candidates = candidates
        all_candidates = candidates
        
        module_clone = module.deepcopy()

        # For each iteration in depth...
        for d in range(self.depth): # TODO: fix this so that we eval the new batch of predictors with the new best followoing predictors
            if self.verbose: print(f"Starting iteration {d}/{self.depth}.")

            latest_scores = []
        
            # Go through our module's predictors
            for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors())):
                candidates_ = latest_candidates[id(p_old)] # Use the most recently generated candidates for evaluation 
                if len(module.predictors()) > 1:
                    candidates_ = all_candidates[id(p_old)] # Unless our program has multiple predictors, in which case we need to reevaluate all prompts with the new prompt(s) for the other predictor(s)   

                # For each candidate
                for c_i, c in enumerate(candidates_):                    
                    # Get the candidate instruction and prefix 
                    instruction, prefix = c.proposed_instruction.strip('"').strip(), c.proposed_prefix_for_output_field.strip('"').strip()

                    # Set this new module with our instruction / prefix 
                    if (hasattr(p_new, 'extended_signature')):
                        *_, last_key = p_new.extended_signature.fields.keys()
                        p_new.extended_signature = p_new.extended_signature \
                            .with_instructions(instruction) \
                            .with_updated_fields(last_key, prefix=prefix)
                    else:
                        *_, last_key = p_new.extended_signature1.fields.keys()
                        p_new.extended_signature1 = p_new.extended_signature1 \
                            .with_instructions(instruction) \
                            .with_updated_fields(last_key, prefix=prefix)
                        *_, last_key = p_new.extended_signature2.fields.keys()
                        p_new.extended_signature2 = p_new.extended_signature2 \
                            .with_instructions(instruction) \
                            .with_updated_fields(last_key, prefix=prefix)

                    # Score the instruction / prefix 
                    if self.verbose: print("----------------")
                    for i,predictor in enumerate(module_clone.predictors()):
                        if self.verbose: print(f"Predictor {i}")
                        self._print_signature(predictor)
                    if self.verbose: print(f"At Depth {d}/{self.depth}, Evaluating Prompt Candidate #{c_i}/{len(candidates_)} for Predictor {p_i} of {len(module.predictors())}.")
                    score = evaluate(module_clone, devset=devset, **eval_kwargs)
                    if self.verbose and self.prompt_model: print(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1
                    if self.verbose: print("----------------")

                    replace_entry = True
                    if self.verbose: print(f"(instruction, prefix) {(instruction, prefix)}")
                    # if verbose: print(f"evaluated_candidates[id(p_old)] {evaluated_candidates[id(p_old)]}")
                    if ((instruction, prefix) in evaluated_candidates[id(p_old)]):
                        # if verbose: print(f"if evaluated_candidates[id(p_old)][(instruction, prefix)] {evaluated_candidates[id(p_old)][(instruction, prefix)]}")
                        if evaluated_candidates[id(p_old)][(instruction, prefix)]["score"] >= score:
                            replace_entry = False

                    if replace_entry:
                        # Add it to our evaluated candidates list
                        evaluated_candidates[id(p_old)][(instruction, prefix)] = {
                            "score": score,
                            "program": module_clone.deepcopy(),
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d,
                        }
                    
                    if (len(candidates_)-self.breadth <= c_i):
                        latest_scores.append(score)

                if self.track_stats:
                    results_latest[id(p_old)]["depth"].append(d)
                    results_latest[id(p_old)]["max"].append(max(latest_scores))
                    results_latest[id(p_old)]["average"].append(sum(latest_scores)/len(latest_scores))
                    results_latest[id(p_old)]["min"].append(min(latest_scores))
                    results_latest[id(p_old)]["std"].append(np.std(latest_scores))
                
                # Now that we've evaluated the candidates, set this predictor to the best performing version
                # to ensure the next round of scores reflect the best possible version
                best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate['score'])
                if (hasattr(p_new, 'extended_signature')):
                    *_, last_key = p_old.extended_signature.fields.keys()
                    p_new.extended_signature = p_new.extended_signature \
                        .with_instructions(best_candidate["instruction"]) \
                        .with_updated_fields(last_key, prefix=best_candidate["prefix"])
                else:
                    *_, last_key1 = p_old.extended_signature1.fields.keys()
                    p_new.extended_signature1 = p_new.extended_signature \
                        .with_instructions(best_candidate["instruction"]) \
                        .with_updated_fields(last_key1, prefix=best_candidate["prefix"])
                    *_, last_key2 = p_old.extended_signature2.fields.keys()
                    p_new.extended_signature2 = p_new.extended_signature \
                        .with_instructions(best_candidate["instruction"]) \
                        .with_updated_fields(last_key2, prefix=best_candidate["prefix"])
                if self.verbose: print(f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\np: {best_candidate['prefix']}")
                if self.verbose: print("Full predictor with update: ")
                for i,predictor in enumerate(module_clone.predictors()):
                    if self.verbose: print(f"Predictor {i}")
                    self._print_signature(predictor)

            if d == self.depth-1:
                break

            
            new_candidates = {}
            for p_base in module.predictors():
                # Build Few-Shot Example of Optimized Prompts
                attempts = []
                shortest_len = self.breadth
                shortest_len = min(len(evaluated_candidates[id(p_base)]),shortest_len)
                best_predictors = list(evaluated_candidates[id(p_base)].values())

                # best_predictors = evaluated_candidates[id(p_base)].values()[:]
                best_predictors.sort(key=lambda x: x['score'], reverse=True)

                if self.track_stats:
                    scores = [x['score'] for x in best_predictors][:10]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(scores))
                    results_best[id(p_base)]["average"].append(sum(scores)/len(scores))
                    results_best[id(p_base)]["min"].append(min(scores))
                    results_best[id(p_base)]["std"].append(np.std(scores))
                
                for i in range(shortest_len-1,-1,-1):
                    # breakpoint()
                    attempts.append(f'Instruction #{shortest_len-i}: {best_predictors[i]["instruction"]}')
                    attempts.append(f'Prefix #{shortest_len-i}: {best_predictors[i]["prefix"]}')
                    attempts.append(f'Resulting Score #{shortest_len-i}: {best_predictors[i]["score"]}')
            
                # Generate next batch of potential prompts to optimize, with previous attempts as input
                if self.prompt_model: 
                    with dspy.settings.context(lm=self.prompt_model):
                        instr = dspy.Predict(GenerateInstructionGivenAttempts, n=self.breadth, temperature=self.init_temperature)(attempted_instructions=attempts)
                else:
                    instr = dspy.Predict(GenerateInstructionGivenAttempts, n=self.breadth, temperature=self.init_temperature)(attempted_instructions=attempts)

                if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")
                # Get candidates for each predictor
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(instr.completions.proposed_prefix_for_output_field)

            if self.verbose and self.prompt_model: print(f"{self.prompt_model.inspect_history(n=1)}")
            latest_candidates = new_candidates
        
        candidates = []
        for predictor in module.predictors():
            candidates.extend(list(evaluated_candidates[id(predictor)].values()))

            if self.track_stats:
                best_predictors = list(evaluated_candidates[id(predictor)].values())
                best_predictors.sort(key=lambda x: x['score'], reverse=True)

                scores = [x['score'] for x in best_predictors][:10]
                results_best[id(predictor)]["depth"].append(d)
                results_best[id(predictor)]["max"].append(max(scores))
                results_best[id(predictor)]["average"].append(sum(scores)/len(scores))
                results_best[id(predictor)]["min"].append(min(scores))
                results_best[id(predictor)]["std"].append(np.std(scores))

        # if verbose: print(f"candidates: {candidates}")
        candidates.sort(key=lambda x: x['score'], reverse=True)

        candidates = self._drop_duplicates(candidates)

        best_program = candidates[0]["program"]
        best_program.candidate_programs = candidates
        best_program.total_calls = total_calls
        if self.track_stats:
            best_program.results_best = results_best
            best_program.results_latest = results_latest

        return best_program