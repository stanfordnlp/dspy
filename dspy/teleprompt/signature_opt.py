import dsp
import dspy
import pickle

from dspy.teleprompt.teleprompt import Teleprompter
from dspy.signatures import Signature

from dspy.evaluate.evaluate import Evaluate

# Define teleprompter 
class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class GenerateInstructionGivenAttempts(Signature):
        """You are an experienced instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality. Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

        attempted_instructions = dspy.InputField(format=dsp.passages2text)
        proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
        proposed_prefix_for_output_field = dspy.OutputField(desc="The string at the end of the prompt, which will help the model start solving the task")

class SignatureOptimizer(Teleprompter):
    def __init__(self, metric=None, breadth=10, depth=3, init_temperature=1.4, prompt_model="gpt-3.5-turbo-1106", output_file=None, verbose=False):
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.output_file = output_file
        self.verbose = verbose

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors()):
            if not p1.extended_signature.instructions == p2.extended_signature.instructions:
                return False
            if not p1.extended_signature.fields[-1] == p2.extended_signature.fields[-1]:
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
    
    def compile(self, student, *, devset, eval_kwargs):
        """student is a program that needs to be optimized, note that it may be zero-shot or already pre-optimized for demos != []"""
        module = student.deepcopy()
        evaluate = Evaluate(devset=devset, metric=self.metric, **eval_kwargs)
        total_calls = 0

        candidates = {}
        evaluated_candidates = {}

        # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
        for predictor in module.predictors():
            basic_instruction = predictor.extended_signature.instructions
            basic_prefix = predictor.extended_signature.fields[-1].name
            with dspy.settings.context(lm=self.prompt_model):
                instruct = dspy.Predict(BasicGenerateInstruction, n=self.breadth-1, temperature=self.init_temperature)(basic_instruction=basic_instruction)
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = []

        last_best_programs = []
        seen = set()
        latest_candidates = candidates
        all_candidates = candidates
        
        module_clone = module.deepcopy()

        # For each iteration in depth...
        for d in range(self.depth): # TODO: fix this so that we eval the new batch of predictors with the new best followoing predictors
            if (self.verbose):
                print(f"Starting iteration {d}/{self.depth}.")
        
            # Go through our module's predictors
            for p_old, p_new in zip(module.predictors(), module_clone.predictors()):
                
                # Get the latest round of prompt candidates for current predictor
                candidates_ = all_candidates[id(p_old)]

                # For each candidate
                for c in candidates_:
                    
                    # Get the candidate instruction and prefix 
                    instruction, prefix = c.proposed_instruction.strip('"').strip(), c.proposed_prefix_for_output_field.strip('"').strip()

                    # Set this new module with our instruction / prefix 
                    p_new.extended_signature.instructions = instruction
                    p_new.extended_signature.fields[-1] = p_new.extended_signature.fields[-1]._replace(name=prefix)

                    # Score the instruction / prefix 
                    if (self.verbose):
                        print(f"----------------")
                        for i,predictor in enumerate(module_clone.predictors()):
                            print(f"Predictor {i}")
                            print(f"i: {predictor.extended_signature.instructions}")
                            print(f"p: {predictor.extended_signature.fields[-1].name}")
                            print()
                    score = evaluate(module_clone, devset=devset, **eval_kwargs)
                    total_calls += 1
                    if (self.verbose):
                        print(f"----------------")

                    # Add it to our evaluated candidates list
                    evaluated_candidates[id(p_old)].append({
                        "score": score,
                        "program": module_clone.deepcopy(),
                        "instruction": instruction,
                        "prefix": prefix,
                        "depth": d
                    })
                
                # Now that we've evaluated the candidates, set this predictor to the best performing version
                # to ensure the next round of scores reflect the best possible version
                best_candidate = max(evaluated_candidates[id(p_old)], key=lambda candidate: candidate['score'])
                p_new.extended_signature.instructions = best_candidate["instruction"]
                p_new.extended_signature.fields[-1] = p_new.extended_signature.fields[-1]._replace(name=best_candidate["prefix"])
                if (self.verbose):
                    print(f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\np: {best_candidate['prefix']}")
                    print(f"Full predictor with update: ")
                    for i,predictor in enumerate(module_clone.predictors()):
                        print(f"Predictor {i}")
                        print(f"i: {predictor.extended_signature.instructions}")
                        print(f"p: {predictor.extended_signature.fields[-1].name}")
                        print()

            for predictor_candidate_list in evaluated_candidates.values():
                predictor_candidate_list.sort(key=lambda x: x['score'], reverse=True)

            if d == self.depth-1:
                break

            # Build Few-Shot Example of Optimized Prompts
            attempts = {}
            shortest_len = self.breadth
            for p in module.predictors():
                attempts[id(p)] = [] # Initialize as empty list
                shortest_len = min(len(evaluated_candidates[id(p)]),shortest_len)
                
            for i in range(shortest_len-1,-1,-1):
                for p_base in module.predictors():
                    # breakpoint()
                    attempts[id(p_base)].append(f'Instruction #{shortest_len-i}: {evaluated_candidates[id(p_base)][i]["instruction"]}')
                    attempts[id(p_base)].append(f'Prefix #{shortest_len-i}: {evaluated_candidates[id(p_base)][i]["prefix"]}')
                    attempts[id(p_base)].append(f'Resulting Score #{shortest_len-i}: {evaluated_candidates[id(p_base)][i]["score"]}')
            
            # Generate next batch of potential prompts to optimize, with previous attempts as input
            new_candidates = {}
            for predictor in module.predictors():
                with dspy.settings.context(lm=self.prompt_model):
                    instr = dspy.Predict(GenerateInstructionGivenAttempts, n=self.breadth, temperature=self.init_temperature)(attempted_instructions=attempts[id(predictor)])
                if (self.verbose):
                    print(f"{self.prompt_model.inspect_history(n=1)}")
                # Get candidates for each predictor
                new_candidates[id(predictor)] = instr.completions
                all_candidates[id(predictor)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(predictor)].proposed_prefix_for_output_field.extend(instr.completions.proposed_prefix_for_output_field)

            latest_candidates = new_candidates

        for candidate_list in evaluated_candidates.values():
            candidate_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Save the evaluated candidates using pickle
        if (self.output_file is not None):
            with open(self.output_dir, 'wb') as file:
                pickle.dump(evaluated_candidates, file)

        best_program = evaluated_candidates[id(p_old)][0]["program"] # Uses python loose scoping to get the best candidate from the last round

        candidates = []
        for predictor in module.predictors():
            candidates.extend(evaluated_candidates[id(predictor)])

        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = self._drop_duplicates(candidates)

        best_program.candidate_programs = candidates
        best_program.total_calls = total_calls

        return best_program