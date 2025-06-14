
import os
import traceback
from typing import Union
import json
import wandb

from .instruction_proposal import ProposeNewInstructionModule

from dspy.teleprompt import Teleprompter
from dspy.clients import LM
from dspy.primitives import Module
from dspy.primitives import Example
from dspy.evaluate import Evaluate
from dspy.dsp.utils.settings import settings

from .gepa_utils import (
    calculate_aggregate_trainval_scores,
    idxmax,
    select_next_candidate_to_update,
    capture_module_trace_with_feedback,
    log_detailed_metrics_after_discovering_new_program,
    initialize_gepa_state,
    initialize_wandb,
    run_simba_on_all_gepa_programs_and_update_gepa_state,
    find_dominator_programs,
    GEPAState
)

from .merge_programs import (
    sample_and_attempt_merge_programs_by_common_predictors
)

class GEPA(Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: dict[str, callable],
        knowledgebase_qe,
        metric: callable,
        logger,
        run_dir: str,
        run_linearized_gepa: bool=True,
        num_threads=None,
        num_iters=30,
        failure_score=0,
        perfect_score=1,
        teacher_lm: LM = None,
        use_wandb: bool = False,
        wandb_api_key: str = None,
        max_evals_per_trainval_instance=30,
        seed=0,
        skip_perfect_score=True,
        use_SIMBA=False,
        SIMBA_max_steps=2,
        max_SIMBA_invocation=3,
        SIMBA_metric=None,
        use_merge=False,
        max_merge_invocations=5,
    ):
        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.knowledgebase_qe = knowledgebase_qe
        self.metric_fn = metric
        self.logger = logger
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa
        self.num_threads = num_threads
        self.num_iters = num_iters
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.teacher_lm = teacher_lm
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.max_evals_per_trainval_instance = max_evals_per_trainval_instance
        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_SIMBA = use_SIMBA
        self.SIMBA_max_steps = SIMBA_max_steps
        self.max_SIMBA_invocation = max_SIMBA_invocation
        self.SIMBA_metric = SIMBA_metric
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        if use_SIMBA:
            assert SIMBA_metric is not None, "SIMBA_metric must be provided if use_SIMBA is True"

    def compile(
        self, student, trainset, valset,
    ):
        gepa_state = self.gepa(
            base_dspy_program=student,
            named_predictor_to_feedback_fn_map=self.named_predictor_to_feedback_fn_map,
            trainset=trainset,
            knowledgebase_qe=self.knowledgebase_qe,
            metric_fn=self.metric_fn,
            logger=self.logger,
            run_dir=self.run_dir,
            valset=valset,
            run_linearized_gepa=self.run_linearized_gepa,
            num_threads=self.num_threads,
            num_iters=self.num_iters,
            failure_score=self.failure_score,
            perfect_score=self.perfect_score,
            teacher_lm=self.teacher_lm,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            max_evals_per_trainval_instance=self.max_evals_per_trainval_instance,
            seed=self.seed,
            me_sample_size=400,
            me_elite_ratio=0.4, 
            me_iterations=50, 
            me_reversed=False,
            skip_perfect_score=self.skip_perfect_score,
            use_SIMBA=self.use_SIMBA,
            SIMBA_max_steps=self.SIMBA_max_steps,
            max_SIMBA_invocation=self.max_SIMBA_invocation,
            SIMBA_metric=self.SIMBA_metric,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations
        )

        agg_trainval_scores = calculate_aggregate_trainval_scores(
            prog_ids=range(len(gepa_state.program_candidates)),
            train_scores=gepa_state.program_full_scores,
            val_scores=gepa_state.program_full_scores_val_set,
            trainset_len=len(trainset),
            valset_len=len(valset)
        )

        best_prog_idx = idxmax(agg_trainval_scores)
        best_prog = gepa_state.program_candidates[best_prog_idx]
        return best_prog

    def gepa(
        self,
        base_dspy_program: Module, 
        named_predictor_to_feedback_fn_map: dict[str, callable],
        trainset: list[Example],
        knowledgebase_qe, 
        metric_fn: callable,
        logger,
        run_dir: str,
        valset: list[Example]=None,
        run_linearized_gepa: bool=True,
        gepa_state_to_use: Union[GEPAState, None]=None,
        num_threads=None,
        num_iters=5,
        failure_score=0,
        perfect_score=1,
        teacher_lm: LM = None,
        use_wandb: bool = False,
        wandb_api_key: str = None,
        max_evals_per_trainval_instance=30,
        use_SIMBA=False,
        SIMBA_max_steps=2,
        max_SIMBA_invocation=3,
        SIMBA_metric=None,
        use_merge=False,
        max_merge_invocations=5,
        seed=0,
        skip_perfect_score=False
    ):
        if use_wandb:
            wandb_run = initialize_wandb(wandb_api_key=wandb_api_key, run_dir=run_dir)

        if num_threads is None:
            num_threads = os.cpu_count()

        trainset_evaluator = Evaluate(
            devset=trainset,
            metric=metric_fn,
            num_threads=num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=failure_score,
            provide_traceback=True
        )

        if valset is None:
            valset = trainset

        valset_evaluator = Evaluate(
            devset=valset,
            metric=metric_fn,
            num_threads=num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=failure_score,
            provide_traceback=True
        )

        gepa_state = initialize_gepa_state(
            gepa_state_to_use=gepa_state_to_use,
            run_dir=run_dir,
            logger=logger,
            base_dspy_program=base_dspy_program,
            trainset_evaluator=trainset_evaluator,
            valset_evaluator=valset_evaluator,
            seed=seed,
            run_linearized_gepa=run_linearized_gepa
        )

        assert len(gepa_state.pareto_front) == len(trainset)

        if use_wandb:
            assert gepa_state.i + 1 == 0
            wandb.log({
                "base_program_full_trainset_score": gepa_state.program_full_scores[0],
                "base_program_full_valset_score": gepa_state.program_full_scores_val_set[0],
                "iteration": gepa_state.i+1,
            })
        logger.log(f"Iteration {gepa_state.i+1}: Base program full trainset score: {gepa_state.program_full_scores[0]}")
        logger.log(f"Iteration {gepa_state.i+1}: Base program full valset score: {gepa_state.program_full_scores_val_set[0]}")

        num_SIMBA_invocations = 0

        merges_due = 0

        while gepa_state.num_full_ds_evals < num_iters and gepa_state.total_num_evals_per_trainval_instance < max_evals_per_trainval_instance:
            try:
                gepa_state.save(run_dir)
                gepa_state.i += 1

                curr_prog_id = select_next_candidate_to_update(gepa_state, trainset, valset)
                curr_prog = gepa_state.program_candidates[curr_prog_id]
                
                predictor_to_update_id = gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id]
                gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id] = (predictor_to_update_id + 1) % len(gepa_state.list_of_named_predictors)
                predictor_name_to_update = gepa_state.list_of_named_predictors[predictor_to_update_id]
                if predictor_name_to_update not in named_predictor_to_feedback_fn_map:
                    logger.log(f"Iteration {gepa_state.i+1}: Predictor {predictor_name_to_update} not in feedback map, skipping")
                    gepa_state.i -= 1
                    continue

                logger.log(f"Iteration {gepa_state.i+1}: Selected program candidate {curr_prog_id} with base score: {gepa_state.program_full_scores[curr_prog_id]}")
                logger.log(f"Iteration {gepa_state.i+1}: Updating predictor {predictor_name_to_update}")

                if use_wandb:
                    wandb.log({
                        "iteration": gepa_state.i+1,
                        "selected_program_candidate": curr_prog_id,
                        "predictor_to_update_id": predictor_to_update_id,
                    }, step=gepa_state.i+1)
                
                feedback_func = named_predictor_to_feedback_fn_map[predictor_name_to_update]
                module = None
                for m in curr_prog.named_predictors():
                    if m[0] == predictor_name_to_update:
                        module = m[1]
                        break
                assert module is not None

                subsample_ids = gepa_state.rng1.sample(range(len(trainset)), k=min(3, len(trainset)))

                dataset_with_feedback, subsample_score = capture_module_trace_with_feedback(
                    module, 
                    curr_prog, 
                    [trainset[i] for i in subsample_ids], 
                    metric_fn, 
                    logger, 
                    gepa_state,
                    skip_perfect_score,
                    perfect_score,
                    feedback_func
                )

                if dataset_with_feedback is None or subsample_score is None:
                    logger.log(f"Iteration {gepa_state.i+1}: No feedback samples, skipping")
                    gepa_state.i -= 1
                    continue

                if use_wandb:
                    wandb.log({
                        "subsample_score": subsample_score,
                    }, step=gepa_state.i+1)

                instruction_propose_module = ProposeNewInstructionModule(
                    base_program=module, 
                    instruction_lm=self.teacher_lm or settings.lm or curr_prog.get_lm(),
                    dataset_with_feedback=dataset_with_feedback, 
                    knowledgebase_qe=knowledgebase_qe)

                try:
                    output = instruction_propose_module.compile()
                    with open(os.path.join(run_dir, "instruction_proposer_inpouts.jsonl"), 'a') as f:
                        f.write(json.dumps(output, default=lambda x: {**x}) + "\n")
                    new_instruction = output['new_instruction']
                    module_output = output['module_output']
                    kb_info = output['kb_info']
                except Exception as e:
                    logger.log(f"Iteration {gepa_state.i+1}: Exception during instruction proposal: {e}")
                    logger.log(traceback.format_exc())
                    gepa_state.i -= 1
                    continue
                logger.log(f"Iteration {gepa_state.i+1}: Info retrieved from knowledge base: {kb_info}")
                logger.log(f"Iteration {gepa_state.i+1}: Proposed new instruction: {new_instruction}")

                curr_prog_lm = curr_prog.get_lm()
                new_program = curr_prog.deepcopy()
                new_program.set_lm(curr_prog_lm)
                new_program.named_predictors()[predictor_to_update_id][1].signature = new_program.named_predictors()[predictor_to_update_id][1].signature.with_instructions(new_instruction)

                subsample_evaluator_args = {**trainset_evaluator.__dict__}
                subsample_evaluator_args['devset'] = [trainset[i] for i in subsample_ids]
                subsample_evaluator_args['return_outputs'] = True
                subsample_evaluator_args['return_all_scores'] = True
                subsample_evaluator = Evaluate(**subsample_evaluator_args)
                new_subsample_score = sum(subsample_evaluator(new_program)[2])

                gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / (len(trainset) + len(valset) if valset is not None else 0)

                logger.log(f"Iteration {gepa_state.i+1}: New subsample score: {new_subsample_score}")
                if use_wandb:
                    wandb.log({
                        "new_subsample_score": new_subsample_score,
                    }, step=gepa_state.i+1)
                
                if new_subsample_score <= subsample_score:
                    logger.log(f"Iteration {gepa_state.i+1}: New subsample score is not better, skipping")
                    continue
                
                logger.log(f"Iteration {gepa_state.i+1}: New subsample score is better, going from {subsample_score} to {new_subsample_score}, updating program candidate!")

                # Calculate metrics for new program and update gepa state
                trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_program)
                valset_score, valset_outputs, valset_subscores = valset_evaluator(new_program)

                # We have run one full eval of the new program on train set and val set
                gepa_state.num_full_ds_evals += 1
                gepa_state.total_num_evals_per_trainval_instance += 1

                new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
                    parent_program_idx=curr_prog_id,
                    new_program=new_program,
                    trainset_score=trainset_score,
                    trainset_outputs=trainset_outputs,
                    trainset_subscores=trainset_subscores,
                    valset_score=valset_score,
                    valset_outputs=valset_outputs,
                    valset_subscores=valset_subscores,
                    run_dir=run_dir
                )

                if new_program_idx == linear_pareto_front_program_idx:
                    logger.log(f"Iteration {gepa_state.i+1}: New program is on the linear pareto front")

                log_detailed_metrics_after_discovering_new_program(
                    logger=logger,
                    gepa_state=gepa_state,
                    valset_score=valset_score,
                    new_prog_all_scores=trainset_subscores,
                    full_score=trainset_score,
                    new_program_idx=new_program_idx,
                    valset_subscores=valset_subscores,
                    new_instruction=new_instruction,
                    use_wandb=use_wandb,
                    linear_pareto_front_program_idx=linear_pareto_front_program_idx
                )

                if new_program_idx == linear_pareto_front_program_idx and use_SIMBA and num_SIMBA_invocations < max_SIMBA_invocation:
                    num_SIMBA_invocations += 1
                    run_simba_on_all_gepa_programs_and_update_gepa_state(
                        SIMBA_metric=SIMBA_metric,
                        gepa_state=gepa_state,
                        trainset_evaluator=trainset_evaluator,
                        valset_evaluator=valset_evaluator,
                        run_dir=run_dir,
                        trainset=trainset,
                        valset=valset,
                        logger=logger,
                        num_threads=num_threads,
                        SIMBA_max_steps=SIMBA_max_steps,
                        use_wandb=use_wandb
                    )

                if use_merge:
                    # We want to evenly distribute the merge invocations across attempts
                    if gepa_state.num_full_ds_evals > 0 and gepa_state.num_full_ds_evals % (num_iters // max_merge_invocations) == 0:
                        merges_due += 1
                    if merges_due > 0:
                        agg_scores = calculate_aggregate_trainval_scores(
                            prog_ids=range(len(gepa_state.program_candidates)),
                            train_scores=gepa_state.program_full_scores,
                            val_scores=gepa_state.program_full_scores_val_set,
                            trainset_len=len(trainset),
                            valset_len=len(valset)
                        )

                        merge_candidates = find_dominator_programs(gepa_state, agg_scores)
                        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
                            gepa_state=gepa_state,
                            agg_scores=agg_scores,
                            rng=gepa_state.rng1,
                            merge_candidates=merge_candidates,
                        )

                        if merge_output[0]:
                            success, new_program, id1, id2, ancestor = merge_output
                            logger.log(f"Iteration {gepa_state.i+1}: Merged programs {id1} and {id2} via ancestor {ancestor}")
                            
                            # Calculate metrics for new program and update gepa state
                            trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_program)
                            valset_score, valset_outputs, valset_subscores = valset_evaluator(new_program)

                            # We have run one full eval of the new program on train set and val set
                            gepa_state.num_full_ds_evals += 1
                            gepa_state.total_num_evals_per_trainval_instance += 1

                            new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
                                parent_program_idx=id1, # TODO: Handle this better. Mark both parents
                                new_program=new_program,
                                trainset_score=trainset_score,
                                trainset_outputs=trainset_outputs,
                                trainset_subscores=trainset_subscores,
                                valset_score=valset_score,
                                valset_outputs=valset_outputs,
                                valset_subscores=valset_subscores,
                                run_dir=run_dir
                            )

                            if new_program_idx == linear_pareto_front_program_idx:
                                logger.log(f"Iteration {gepa_state.i+1}: New program is on the linear pareto front")

                            log_detailed_metrics_after_discovering_new_program(
                                logger=logger,
                                gepa_state=gepa_state,
                                valset_score=valset_score,
                                new_prog_all_scores=trainset_subscores,
                                full_score=trainset_score,
                                new_program_idx=new_program_idx,
                                valset_subscores=valset_subscores,
                                new_instruction="Merged program",
                                use_wandb=use_wandb,
                                linear_pareto_front_program_idx=linear_pareto_front_program_idx
                            )

                            merges_due -= 1
                        else:
                            logger.log(f"Iteration {gepa_state.i+1}: No merge candidates found")
                            merges_due += 1
                    
            except Exception as e:
                logger.log(f"Iteration {gepa_state.i+1}: Exception during optimization: {e}")
                logger.log(traceback.format_exc())
                continue

        if use_SIMBA: # num_SIMBA_invocations < max_SIMBA_invocation and 
            run_simba_on_all_gepa_programs_and_update_gepa_state(
                SIMBA_metric=SIMBA_metric,
                gepa_state=gepa_state,
                trainset_evaluator=trainset_evaluator,
                valset_evaluator=valset_evaluator,
                run_dir=run_dir,
                trainset=trainset,
                valset=valset,
                logger=logger,
                num_threads=num_threads,
                SIMBA_max_steps=8,
                use_wandb=use_wandb
            )
        
        gepa_state.save(run_dir)

        return gepa_state
