from copy import deepcopy
import os
import traceback
import random
import itertools
import json

import wandb

from .entropy_utils import remove_dominated_programs

from dspy.primitives import Module
from dspy.primitives import Example
from dspy.primitives import Prediction
from dspy.utils.saving import load as dspy_load
from dspy.dsp.utils.settings import settings
from dspy.evaluate import Evaluate
from dspy.teleprompt.simba import SIMBA

class GEPAState:
    program_candidates: list[Module]
    parent_program_for_candidate: list[int | None]

    program_full_scores: list[float]
    program_full_scores_val_set: list[float]
    
    pareto_front: list[float]
    
    program_at_pareto_front: list[set[int]]
    program_at_pareto_front_valset: list[set[int]]
    pareto_front_score_at_iter: list[float]

    prog_candidate_train_subscores: list[list[float]]
    prog_candidate_val_subscores: list[list[float]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    total_rollouts: int
    num_full_ds_evals: int

    total_num_evals_per_trainval_instance: float

    running_linearized_gepa: bool

    rng1: random.Random
    rng2: random.Random

    def __init__(
        self, 
        base_program: Module, 
        base_trainset_eval_output: tuple[float, list[Prediction], list[float]],
        base_valset_eval_output: tuple[float, list[Prediction], list[float]],
        seed: int, 
        run_linearized_gepa: bool=False
    ):
        base_program_full_score = base_trainset_eval_output[0]
        base_pareto_front = base_trainset_eval_output[2]
        valset_base_score = base_valset_eval_output[0]
        base_valset_pareto_front = base_valset_eval_output[2]

        base_program_lm = base_program.get_lm()
        first_program_candidate = base_program.deepcopy()
        first_program_candidate.set_lm(base_program_lm)

        self.program_candidates = [first_program_candidate]
        self.program_full_scores = [base_program_full_score]
        self.program_full_scores_val_set = [valset_base_score]
        self.pareto_front = base_pareto_front
        self.pareto_front_valset = base_valset_pareto_front
        self.parent_program_for_candidate = [None]
        self.program_at_pareto_front = [{0} for _ in range(len(base_pareto_front))]
        self.program_at_pareto_front_valset = [{0} for _ in range(len(base_valset_pareto_front))]
        self.pareto_front_score_at_iter = [sum(base_pareto_front)/len(base_pareto_front)]
        self.num_full_ds_evals = 0
        self.list_of_named_predictors = [k[0] for k in base_program.named_predictors()]
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1
        self.rng1 = random.Random(seed)
        self.rng2 = random.Random(seed+1)
        self.total_rollouts = 0

        self.prog_candidate_train_subscores = [base_trainset_eval_output[2]]
        self.prog_candidate_val_subscores = [base_valset_eval_output[2]]

        self.running_linearized_gepa = run_linearized_gepa

        self.total_num_evals_per_trainval_instance = 1 # We have run one full eval of the base program on train set and val set

    def save(self, run_dir:str):
        for prog_idx, curr_prog in enumerate(self.program_candidates):
            dir_to_save = os.path.join(run_dir, "prog_candidates", str(prog_idx))
            os.makedirs(dir_to_save, exist_ok=True)
            curr_prog_lm = curr_prog.get_lm()
            curr_prog.set_lm(None)
            curr_prog.save(dir_to_save, save_program=True)
            curr_prog.set_lm(curr_prog_lm)
        
        # Save all the other state except programs as pickle
        with open(os.path.join(run_dir, "gepa_state.bin"), 'wb') as f:
            import pickle
            d = {k: v for k, v in self.__dict__.items() if k != 'program_candidates'}
            pickle.dump(d, f)
    
    @staticmethod
    def load(run_dir: str) -> 'GEPAState':
        with open(os.path.join(run_dir, "gepa_state.bin"), 'rb') as f:
            import pickle
            d = pickle.load(f)
        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)
        if not hasattr(state, 'running_linearized_gepa'):
            setattr(state, 'running_linearized_gepa', False)
        state.program_candidates = []
        for i in itertools.count():
            dir_to_load = os.path.join(run_dir, "prog_candidates", str(i))
            if not os.path.exists(dir_to_load):
                break
            prog = dspy_load(dir_to_load)
            state.program_candidates.append(prog)
        
        assert len(state.program_candidates) == len(state.program_full_scores)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        assert len(state.pareto_front) == len(state.program_at_pareto_front)

        return state

    def update_state_with_new_program(self, parent_program_idx, new_program, trainset_score, trainset_outputs, trainset_subscores, valset_score, valset_outputs, valset_subscores, run_dir):
        new_program_idx = len(self.program_candidates)
                
        for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front, trainset_subscores)):
            if new_score > old_score:
                os.makedirs(os.path.join(run_dir, "generated_best_outputs", f"task_{task_idx}"), exist_ok=True)
                with open(os.path.join(run_dir, "generated_best_outputs", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), 'w') as f:
                    json.dump(trainset_outputs[task_idx], f, indent=4, default=json_default)

        self.program_candidates.append(new_program)
        self.named_predictor_id_to_update_next_for_program_candidate.append(self.named_predictor_id_to_update_next_for_program_candidate[parent_program_idx])

        self.pareto_front, self.program_at_pareto_front = update_pareto_front(
            new_prog_all_scores=trainset_subscores, 
            new_program_idx=new_program_idx, 
            new_program_full_score=trainset_score, 
            gepa_state=self
        )
        
        self.program_full_scores.append(trainset_score)
        self.parent_program_for_candidate.append(parent_program_idx)
        self.pareto_front_score_at_iter.append(sum(self.pareto_front)/len(self.pareto_front))

        self.prog_candidate_train_subscores.append(trainset_subscores)
        self.prog_candidate_val_subscores.append(valset_subscores)

        self.program_full_scores_val_set.append(valset_score)
        for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_valset, valset_subscores)):
            if new_score > old_score:
                self.pareto_front_valset[task_idx] = new_score
                self.program_at_pareto_front_valset[task_idx] = {new_program_idx}
                os.makedirs(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}"), exist_ok=True)
                with open(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), 'w') as f:
                    json.dump(valset_outputs[task_idx], f, indent=4, default=json_default)
            elif new_score == old_score:
                self.program_at_pareto_front_valset[task_idx].add(new_program_idx)
        
        assert len(trainset_subscores) == len(self.program_at_pareto_front)
        assert len(valset_subscores) == len(self.program_at_pareto_front_valset)

        train_val_weighted_agg_scores_for_all_programs = calculate_aggregate_trainval_scores(
            prog_ids=range(len(self.program_candidates)),
            train_scores=self.program_full_scores,
            val_scores=self.program_full_scores_val_set,
            trainset_len=len(trainset_subscores),
            valset_len=len(valset_subscores)
        )

        linear_pareto_front_program_idx = idxmax(train_val_weighted_agg_scores_for_all_programs)
        
        return new_program_idx, linear_pareto_front_program_idx

def calculate_aggregate_trainval_scores(prog_ids, train_scores, val_scores, trainset_len, valset_len):
    train_val_weighted_agg_scores_for_all_programs = []
    for prog_idx in prog_ids:
        train_score = train_scores[prog_idx]
        val_score = val_scores[prog_idx]
        train_weight = trainset_len / (trainset_len + valset_len)
        val_weight = valset_len / (trainset_len + valset_len)
        weighted_agg_score = train_weight * train_score + val_weight * val_score
        train_val_weighted_agg_scores_for_all_programs.append(weighted_agg_score)
    
    return train_val_weighted_agg_scores_for_all_programs

def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except:
        return repr(x)

def idxmax(lst):
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

def update_pareto_front(new_prog_all_scores, new_program_idx, new_program_full_score, gepa_state: GEPAState):
    pareto_front, program_at_pareto_front, program_full_scores, run_linearized_gepa = (
        deepcopy(gepa_state.pareto_front),
        deepcopy(gepa_state.program_at_pareto_front), 
        gepa_state.program_full_scores, 
        gepa_state.running_linearized_gepa
    )

    for score_idx, score in enumerate(new_prog_all_scores):
        if score > pareto_front[score_idx]:
            program_at_pareto_front[score_idx] = {new_program_idx}
            pareto_front[score_idx] = score
        elif score == pareto_front[score_idx]:
            program_at_pareto_front[score_idx].add(new_program_idx)
        else:
            pass

    return pareto_front, program_at_pareto_front

def capture_module_trace_with_feedback(
    module: Module,
    full_program: Module,
    evalset: list[Example],
    metric_fn: callable,
    logger,
    gepa_state: GEPAState,
    skip_perfect_score: bool,
    perfect_score: float,
    feedback_func: callable,
):
    # Instrument the forward method to capture input and output
    original_forward = module.forward
    captured = []

    def forward_with_feedback(*args, **kwargs):
        assert len(args) == 0
        inputs = kwargs
        outputs = original_forward(**kwargs)
        captured[-1].append({'inputs': inputs, 'generated_output': outputs})
        return outputs
    
    num_perf = 0
    try:
        module.forward = forward_with_feedback
        subsample_score = 0
        selected_inp_output_pairs = []
        for testcase in evalset:
            captured.append([])
            captured_trace = None
            with settings.context(trace=[]):
                o = full_program(**testcase.inputs())
                captured_trace = settings.trace
            score = metric_fn(testcase, o)
            if score == perfect_score:
                logger.log(f"Iteration {gepa_state.i+1}: Score is 1, skipping")
                num_perf += 1
            subsample_score += score
            if len(captured[-1]) > 0:
                selected_inp_output_pairs.append(gepa_state.rng1.choice(captured[-1]))
                selected_inp_output_pairs[-1]['module_inputs'] = testcase
                selected_inp_output_pairs[-1]['module_outputs'] = o
                selected_inp_output_pairs[-1]['captured_trace'] = captured_trace
    finally:
        module.forward = original_forward
    
    if len(selected_inp_output_pairs) == 0:
        logger.log(f"Iteration {gepa_state.i+1}: No feedback samples, skipping")
        return None, None

    if num_perf == len(evalset) and skip_perfect_score:
        logger.log(f"Iteration {gepa_state.i+1}: All samples are perfect, skipping")
        return None, None

    logger.log(f"Iteration {gepa_state.i+1}: Base program subsample score: {subsample_score}")
    dataset_with_feedback = []
    for d in selected_inp_output_pairs:
        feedback_d = feedback_func(
            predictor_output=d['generated_output'], 
            predictor_inputs=d['inputs'], 
            module_inputs=d['module_inputs'],
            module_outputs=d['module_outputs'],
            captured_trace=d['captured_trace'],
        )
        score, feedback_text = feedback_d["feedback_score"], feedback_d["feedback_text"]
        d['feedback'] = feedback_text
        dataset_with_feedback.append(d)
    
    return dataset_with_feedback, subsample_score

def write_eval_output_to_directory(eval_out, output_dir):
    for task_idx, score in enumerate(eval_out[2]):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), 'w') as f:
            json.dump(eval_out[1][task_idx], f, indent=4, default=json_default)

def initialize_wandb(wandb_api_key: str = None, run_dir: str = None):
    try:
        import wandb
        if wandb_api_key:
            wandb.login(key=wandb_api_key, verify=True)
        else:
            wandb.login()
    except ImportError:
        raise ImportError("wandb is not installed. Please install it or set use_wandb=False.")
    except Exception as e:
        raise RuntimeError(f"Error logging into wandb: {e}")
    
    wandb_run = wandb.init(
        project="gepa",
        dir=os.path.join(run_dir, "wandb"),
        name=run_dir,
    )
    return wandb_run

def initialize_gepa_state(gepa_state_to_use, run_dir, logger, base_dspy_program, trainset_evaluator, valset_evaluator, seed, run_linearized_gepa):
    if gepa_state_to_use is None:
        if os.path.exists(os.path.join(run_dir, "gepa_state.bin")) and os.path.exists(os.path.join(run_dir, "prog_candidates")):
            logger.log("Loading gepa state from run dir")
            gepa_state = GEPAState.load(run_dir)
        else:
            try:
                eval_out = trainset_evaluator(base_dspy_program)
            except Exception as e:
                logger.log(f"Exception during eval: {e}")
                logger.log(traceback.format_exc())
                raise e
            write_eval_output_to_directory(eval_out, os.path.join(run_dir, "generated_best_outputs"))
            
            valset_out = valset_evaluator(base_dspy_program)
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))

            gepa_state = GEPAState(
                base_dspy_program, 
                eval_out,
                valset_out,
                seed,
                run_linearized_gepa
            )
    else:
        gepa_state = gepa_state_to_use

    return gepa_state

def find_dominator_programs(gepa_state: GEPAState, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = (
        gepa_state.program_at_pareto_front_valset + \
        gepa_state.program_at_pareto_front + \
        [
            # {idxmax(train_val_weighted_agg_scores_for_all_programs)}, # Add best aggregate program
            # {idxmax(gepa_state.program_full_scores_val_set)}, # Add best on valset
            # {idxmax(gepa_state.program_full_scores)}, # Add best on trainset
        ] # TODO: Think about whether this should be added or not. Make this configurable.
    )
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    uniq_progs = []
    for front in new_program_at_pareto_front_valset:
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

def select_program_candidate_from_pareto_front(gepa_state: GEPAState, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = (
        gepa_state.program_at_pareto_front_valset + \
        gepa_state.program_at_pareto_front + \
        [
            # {idxmax(train_val_weighted_agg_scores_for_all_programs)}, # Add best aggregate program
            # {idxmax(gepa_state.program_full_scores_val_set)}, # Add best on valset
            # {idxmax(gepa_state.program_full_scores)}, # Add best on trainset
        ] # TODO: Think about whether this should be added or not. Make this configurable.
    )
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset:
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1
    
    sampling_list = [prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)]
    assert len(sampling_list) > 0
    curr_prog_id = gepa_state.rng2.choice(sampling_list)
    return curr_prog_id

def select_next_candidate_to_update(gepa_state: GEPAState, trainset: list[Example], valset: list[Example]):
    train_val_weighted_agg_scores_for_all_programs = calculate_aggregate_trainval_scores(
        prog_ids=range(len(gepa_state.program_candidates)),
        train_scores=gepa_state.program_full_scores,
        val_scores=gepa_state.program_full_scores_val_set,
        trainset_len=len(trainset),
        valset_len=len(valset)
    )
    
    assert len(train_val_weighted_agg_scores_for_all_programs) == len(gepa_state.program_candidates)

    if not gepa_state.running_linearized_gepa:
        curr_prog_id = select_program_candidate_from_pareto_front(gepa_state, train_val_weighted_agg_scores_for_all_programs)
    else:
        curr_prog_id = idxmax(train_val_weighted_agg_scores_for_all_programs)
    
    return curr_prog_id

def log_detailed_metrics_after_discovering_new_program(logger, gepa_state, valset_score, new_prog_all_scores, full_score, new_program_idx, valset_subscores, new_instruction, use_wandb, linear_pareto_front_program_idx, ):
    aggregate_train_val_weighted_scores = calculate_aggregate_trainval_scores(
        prog_ids=range(len(gepa_state.program_candidates)),
        train_scores=gepa_state.program_full_scores,
        val_scores=gepa_state.program_full_scores_val_set,
        trainset_len=len(gepa_state.program_at_pareto_front),
        valset_len=len(gepa_state.program_at_pareto_front_valset)
    )

    best_prog_as_per_agg_score = idxmax(aggregate_train_val_weighted_scores)
    best_prog_as_per_agg_score_valset = idxmax(gepa_state.program_full_scores_val_set)
    best_prog_as_per_agg_score_trainset = idxmax(gepa_state.program_full_scores)

    logger.log(f"Iteration {gepa_state.i+1}: Full valset score for new program: {valset_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Full trainset score for new program: {full_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Full train_val score for new program: {aggregate_train_val_weighted_scores[new_program_idx]}")
    logger.log(f"Iteration {gepa_state.i+1}: Individual valset scores for new program: {valset_subscores}")
    logger.log(f"Iteration {gepa_state.i+1}: Individual trainset scores for new program: {new_prog_all_scores}")
    logger.log(f"Iteration {gepa_state.i+1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: New trainset pareto front scores: {gepa_state.pareto_front}")
    logger.log(f"Iteration {gepa_state.i+1}: Full valset pareto front score: {sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset)}")
    logger.log(f"Iteration {gepa_state.i+1}: Full trainset pareto front score: {sum(gepa_state.pareto_front)/len(gepa_state.pareto_front)}")
    logger.log(f"Iteration {gepa_state.i+1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Updated trainset pareto front programs: {gepa_state.program_at_pareto_front}")
    logger.log(f"Iteration {gepa_state.i+1}: Best trainset aggregate score so far: {max(gepa_state.program_full_scores)}")
    logger.log(f"Iteration {gepa_state.i+1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on train_val: {best_prog_as_per_agg_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on valset: {best_prog_as_per_agg_score_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on trainset: {best_prog_as_per_agg_score_trainset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on trainset: {gepa_state.program_full_scores[best_prog_as_per_agg_score_trainset]}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on valset: {gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset]}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on train_val: {aggregate_train_val_weighted_scores[best_prog_as_per_agg_score]}")
    logger.log(f"Iteration {gepa_state.i+1}: Linear pareto front program index: {linear_pareto_front_program_idx}")
    logger.log(f"Iteration {gepa_state.i+1}: New program candidate index: {new_program_idx}")

    if use_wandb:
        wandb.log({
            "iteration": gepa_state.i+1,
            "full_trainset_score": full_score,
            "full_valset_score": valset_score,
            "full_train_val_score": aggregate_train_val_weighted_scores[new_program_idx],
            "new_instruction": new_instruction,
            "new_program_idx": new_program_idx,
            "valset_pareto_front_scores": gepa_state.pareto_front_valset,
            "trainset_pareto_front_scores": gepa_state.pareto_front,
            "individual_valset_score_new_program": valset_subscores,
            "individual_trainset_score_new_program": new_prog_all_scores,
            "trainset_pareto_front_agg": sum(gepa_state.pareto_front)/len(gepa_state.pareto_front),
            "valset_pareto_front_agg": sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset),
            "trainset_pareto_front_programs": gepa_state.program_at_pareto_front,
            "valset_pareto_front_programs": gepa_state.program_at_pareto_front_valset,
            "best_trainset_agg_score": max(gepa_state.program_full_scores),
            "best_valset_agg_score": max(gepa_state.program_full_scores_val_set),
            "linear_pareto_front_program_idx": linear_pareto_front_program_idx,

            "best_program_as_per_agg_score": best_prog_as_per_agg_score,
            "best_program_as_per_agg_score_valset": best_prog_as_per_agg_score_valset,
            "best_program_as_per_agg_score_trainset": best_prog_as_per_agg_score_trainset,

            "best_score_on_trainset": gepa_state.program_full_scores[best_prog_as_per_agg_score_trainset],
            "best_score_on_valset": gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset],
            "best_score_on_train_val": aggregate_train_val_weighted_scores[best_prog_as_per_agg_score],
        }, step=gepa_state.i+1)

def run_simba_on_one_program_and_update_gepa_state(
    SIMBA_metric: callable,
    gepa_state: GEPAState,
    base_simba_program: Module,
    base_simba_progidx_in_gepa_state: int,
    trainset_evaluator: Evaluate,
    valset_evaluator: Evaluate,
    run_dir: str,
    trainset: list[Example],
    valset: list[Example],
    logger,
    num_threads=None,
    SIMBA_max_steps=2,
    use_wandb: bool = False,
):
    try:
        logger.log(f"Iteration {gepa_state.i+1}: Running SIMBA")
        simba_opt = SIMBA(max_steps=SIMBA_max_steps, num_threads=num_threads, metric=SIMBA_metric)
        optimized_program = simba_opt.compile(
            base_simba_program,
            trainset= trainset + valset,
        )

        SIMBA_candidate_programs = optimized_program.candidate_programs
        del optimized_program.candidate_programs

        logger.log(f"Iteration {gepa_state.i+1}: Obtained SIMBA candidate programs: {len(SIMBA_candidate_programs)}")
        for new_simba_program_candidate in SIMBA_candidate_programs:
            gepa_state.i += 1 # We will log for a new step now
            trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_simba_program_candidate)
            valset_score, valset_outputs, valset_subscores = valset_evaluator(new_simba_program_candidate)

            simba_new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
                parent_program_idx=base_simba_progidx_in_gepa_state,
                new_program=new_simba_program_candidate,
                trainset_score=trainset_score,
                trainset_outputs=trainset_outputs,
                trainset_subscores=trainset_subscores,
                valset_score=valset_score,
                valset_outputs=valset_outputs,
                valset_subscores=valset_subscores,
                run_dir=run_dir
            )

            if simba_new_program_idx == linear_pareto_front_program_idx:
                logger.log(f"Iteration {gepa_state.i+1}: New SIMBA program is on the linear pareto front")

            log_detailed_metrics_after_discovering_new_program(
                logger=logger,
                gepa_state=gepa_state,
                valset_score=valset_score,
                new_prog_all_scores=trainset_subscores,
                full_score=trainset_score,
                new_program_idx=simba_new_program_idx,
                valset_subscores=valset_subscores,
                new_instruction="SIMBA Generated Program",
                use_wandb=use_wandb,
                linear_pareto_front_program_idx=linear_pareto_front_program_idx
            )
    except Exception as e:
        logger.log(f"Iteration {gepa_state.i+1}: Exception during SIMBA: {e}")
        logger.log(traceback.format_exc())

def run_simba_on_all_gepa_programs_and_update_gepa_state(
    SIMBA_metric: callable,
    gepa_state: GEPAState,
    trainset_evaluator: Evaluate,
    valset_evaluator: Evaluate,
    run_dir: str,
    trainset: list[Example],
    valset: list[Example],
    logger,
    num_threads=None,
    SIMBA_max_steps=2,
    use_wandb: bool = False,
):
    try:
        logger.log(f"Iteration {gepa_state.i+1}: Running SIMBA")
        simba_opt = SIMBA(max_steps=SIMBA_max_steps, num_threads=num_threads, metric=SIMBA_metric)

        optimized_program = simba_opt.compile(
            student=None,
            trainset= trainset + valset,
            student_progs=gepa_state.program_candidates,
            student_scores=[
                gepa_state.prog_candidate_train_subscores[prog_idx] + gepa_state.prog_candidate_val_subscores[prog_idx] 
                for prog_idx in range(len(gepa_state.program_candidates))
            ],
        )

        SIMBA_candidate_programs = optimized_program.candidate_programs
        del optimized_program.candidate_programs

        logger.log(f"Iteration {gepa_state.i+1}: Obtained SIMBA candidate programs: {len(SIMBA_candidate_programs)}")
        for new_simba_program_candidate in SIMBA_candidate_programs:
            gepa_state.i += 1 # We will log for a new step now
            trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_simba_program_candidate)
            valset_score, valset_outputs, valset_subscores = valset_evaluator(new_simba_program_candidate)

            simba_new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
                # TODO: We should hook into SIMBA to identify which program it is based on. For now we are using 0
                parent_program_idx=0,
                new_program=new_simba_program_candidate,
                trainset_score=trainset_score,
                trainset_outputs=trainset_outputs,
                trainset_subscores=trainset_subscores,
                valset_score=valset_score,
                valset_outputs=valset_outputs,
                valset_subscores=valset_subscores,
                run_dir=run_dir
            )

            if simba_new_program_idx == linear_pareto_front_program_idx:
                logger.log(f"Iteration {gepa_state.i+1}: New SIMBA program is on the linear pareto front")

            log_detailed_metrics_after_discovering_new_program(
                logger=logger,
                gepa_state=gepa_state,
                valset_score=valset_score,
                new_prog_all_scores=trainset_subscores,
                full_score=trainset_score,
                new_program_idx=simba_new_program_idx,
                valset_subscores=valset_subscores,
                new_instruction="SIMBA Generated Program",
                use_wandb=use_wandb,
                linear_pareto_front_program_idx=linear_pareto_front_program_idx
            )
    except Exception as e:
        logger.log(f"Iteration {gepa_state.i+1}: Exception during SIMBA: {e}")
        logger.log(traceback.format_exc())