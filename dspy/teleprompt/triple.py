import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_n_fewshot_demo_sets,
    get_signature,
    set_signature,
)


class Triple(Teleprompter):
    def __init__(
        self,
        metric,
        num_instruction_candidates=8,
        num_demo_candidates=4,
        metric_threshhold=None,
        algorithm="SH",
        seed=42,
        prompt_model=None,
        num_data_per_round=2,
        include_non_bootstrapped=False,
        include_labeled_demos=False,
    ):
        super().__init__()
        self.metric = metric
        self.num_instruction_candidates = num_instruction_candidates
        self.metric_threshold = metric_threshhold
        self.algorithm = algorithm
        self.seed = seed
        self.prompt_model = prompt_model
        self.num_data_per_round = num_data_per_round
        self.num_demo_candidates = num_demo_candidates
        self.include_non_bootstrapped = include_non_bootstrapped
        self.include_labeled_demos = include_labeled_demos

    def _bootstrap_fewshot_examples(self, program: Any, trainset: List) -> Optional[List]:
        print("BOOTSTRAP FEWSHOT EXAMPLES")

        try:
            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=4,
                trainset=trainset,
                max_labeled_demos=self.num_demo_candidates,
                max_bootstrapped_demos=self.num_demo_candidates,
                metric=self.metric,
                metric_threshold=self.metric_threshold,
                seed=self.seed,
                teacher_settings=None,
                include_non_bootstrapped=self.include_non_bootstrapped,
                labeled_sample=self.include_labeled_demos,
            )
        except Exception as e:
            print(f"Error generating few-shot examples: {e}")
            print("Running without few-shot examples.")
            demo_candidates = None

        return demo_candidates

    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates=None,
        view_data_batch_size=10,
        program_aware_proposer=True,
        data_aware_proposer=True,
        tip_aware_proposer=True,
        fewshot_aware_proposer=True,
    ) -> Dict[int, List[str]]:
        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=self.num_demo_candidates,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=False,
        )

        print("\nProposing instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruction_candidates,
            T=0.5,
            trial_logs={},
        )

        for i, pred in enumerate(program.predictors()):
            print(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = get_signature(pred).instructions
            for j, instruction in enumerate(instruction_candidates[i]):
                print(f"{j}: {instruction}\n")
            print("\n")

        return instruction_candidates

    def _execute_one_instruction_combo(self, program, instruction_candidates, instruction_index, valset):
        for i, predict in enumerate(program.predictors()):
            selected_instruction = instruction_candidates[i][instruction_index[i]]
            updated_signature = get_signature(predict).with_instructions(selected_instruction)
            set_signature(predict, updated_signature)
        scores = []
        data_indices = random.sample(range(len(valset)), self.num_data_per_round)
        for data_index in data_indices:
            selected_data = valset[data_index]
            try:
                pred = program(**selected_data.inputs())
                scores.append(self.metric(selected_data, pred))
            except:
                scores.append(0)
        return np.mean(scores)

    def _execute_one_demo_combo(self, program, demo_candidates, demo_index, valset):
        for i, predict in enumerate(program.predictors()):
            selected_demo = demo_candidates[i][demo_index[i]]
            predict.demos.append(selected_demo)

        scores = []
        data_indices = random.sample(range(len(valset)), self.num_data_per_round)
        for data_index in data_indices:
            selected_data = valset[data_index]
            try:
                pred = program(**selected_data.inputs())
                scores.append(self.metric(selected_data, pred))
            except:
                scores.append(0)

        # Reset the `demos`.
        for i, predict in enumerate(program.predictors()):
            predict.demos = predict.demos[:-1]

        return np.mean(scores)

    def _triple_sh(self, program, instruction_candidates, trainset, valset=None, budget=None):
        total_rounds = math.ceil(math.log(len(instruction_candidates[0]), 2))
        active_instruction_set = instruction_candidates
        for i in range(total_rounds):
            print(
                f"Starting the bandit selection round: {i}, remaining number of instruction candidates: {len(active_instruction_set[0])}"
            )
            num_trials_per_instruction = math.ceil(budget / (len(active_instruction_set[0]) * total_rounds))
            # Make an array [0, 1, 2, ..., 0, 1, 2, ...], the repetition number is num_trials_per_instruction.
            instruction_indices = [
                i for _ in range(num_trials_per_instruction) for i in range(len(active_instruction_set[0]))
            ]

            instruction_indices_sequence_all = []
            for j in range(len(active_instruction_set)):
                # Generate an instruction sequence for each predict.
                instruction_indices_sequence = list(instruction_indices)
                random.shuffle(instruction_indices_sequence)
                instruction_indices_sequence_all.append(instruction_indices_sequence)

            # Transpose so that each element is a sequence of index of the
            # instruction candidate for each predict. E.g, if there are 2
            # predicts and each has 3 candidates, after transpose
            # `instruction_indices_sequence_all` looks like
            # [[0, 1, 1], [2, 1, 0], ...]
            instruction_indices_sequence_all = np.array(instruction_indices_sequence_all).transpose().tolist()

            # The score book is of shape Np * Ni * Nt
            # Np: number of predicts in the DSPy program
            # Ni: number of instructions candidates for each predict
            # Nt: number of trials required for each instruction candidate
            score_book = np.empty(
                (
                    len(active_instruction_set),
                    len(active_instruction_set[0]),
                    0,
                )
            ).tolist()

            valset = valset

            with tqdm(total=len(instruction_indices_sequence_all)) as pbar:
                for instruction_index in instruction_indices_sequence_all:
                    # Execute with random instruction for each module
                    score = self._execute_one_instruction_combo(
                        program,
                        active_instruction_set,
                        instruction_index,
                        valset,
                    )

                    # Update score_book with the result
                    for predict_index, instr_idx in enumerate(instruction_index):
                        score_book[predict_index][instr_idx].append(score)

                    # Manually update the progress bar
                    pbar.update(1)

            # Calculate the average score for each predict-instruction pair.
            score_book = np.array(score_book)
            mean_score = np.mean(score_book, axis=-1)

            print(f"Score book in round {i}: ")
            print(mean_score)

            # Remove the instruction candidates having score in the bottom 50% for each predict.
            num_to_keep = math.ceil(len(active_instruction_set[0]) / 2.0)
            top_k_index = np.argpartition(-mean_score, num_to_keep, axis=1)[:, :num_to_keep]
            top_k_index = top_k_index.tolist()
            new_active_instruction_set = []
            print("Removing the bottom half instruction candidates...")
            for predict_index, instruction_indices in enumerate(top_k_index):
                candidates_to_keep = [
                    active_instruction_set[predict_index][instruction_index]
                    for instruction_index in instruction_indices
                ]
                new_active_instruction_set.append(candidates_to_keep)

            active_instruction_set = new_active_instruction_set
            print(f"Remaining instruction candidates: {active_instruction_set}")

        # Set the instruction to be the last remaining instruction
        for i, predict in enumerate(program.predictors()):
            selected_instruction = active_instruction_set[i][0]
            updated_signature = get_signature(predict).with_instructions(selected_instruction)
            set_signature(predict, updated_signature)

        return program

    def _triple_cr(self, program, instruction_candidates, trainset, valset=None, budget=None):
        num_predicts = len(instruction_candidates)
        num_candidates_per_predict = len(instruction_candidates[0])

        n_p = np.array([[0 for j in range(num_candidates_per_predict)] for i in range(num_predicts)])
        u_p = np.array([[0 for j in range(num_candidates_per_predict)] for i in range(num_predicts)], dtype=np.float32)

        # `n_p_mask` and `u_p_mask` are used to mask out inactive instruction candidates.
        n_p_mask = np.array([[0 for j in range(num_candidates_per_predict)] for i in range(num_predicts)])
        u_p_mask = np.array(
            [[0 for j in range(num_candidates_per_predict)] for i in range(num_predicts)], dtype=np.float32
        )

        active_instruction_candidates_indices = [set(range(num_candidates_per_predict)) for _ in range(num_predicts)]
        inactive_instruction_candidates_indices = [set() for _ in range(num_predicts)]

        with tqdm(total=budget) as pbar:
            for step in range(budget):
                instruction_indices = np.argmin(n_p + n_p_mask, axis=1).tolist()

                print(f"Selected instruction indices: {instruction_indices}")
                print(f"The time of each candidate been seen: {n_p[np.arange(len(n_p)), instruction_indices]}")
                score = self._execute_one_instruction_combo(
                    program, instruction_candidates, instruction_indices, valset
                )

                for predict_index, instruction_index in enumerate(instruction_indices):
                    u_p[predict_index][instruction_index] = (
                        u_p[predict_index][instruction_index] * n_p[predict_index][instruction_index] + score
                    ) / (n_p[predict_index][instruction_index] + 1)
                    n_p[predict_index][instruction_index] += 1

                for predict_index in range(num_predicts):
                    p_prime_index = np.argmin(u_p[predict_index] + u_p_mask[predict_index])

                    sorted_score = sorted(u_p[predict_index] + u_p_mask[predict_index])
                    u_p_smallest = sorted_score[0]
                    u_p_second_smallet = sorted_score[1]

                    delta = u_p_second_smallet - u_p_smallest

                    filter_threshold = (
                        budget
                        - np.sum(n_p[predict_index][list(inactive_instruction_candidates_indices[predict_index])])
                    ) / (
                        math.log(len(active_instruction_candidates_indices[predict_index]))
                        * np.sum(n_p[predict_index][list(active_instruction_candidates_indices[predict_index])])
                    )
                    # The `filter_threshold` cannot be smaller than 0.
                    filter_threshold = max(0, np.sqrt(filter_threshold) - 1)

                    print(
                        f"Evaluate the worst candidate {p_prime_index} for predict {predict_index}, delta score: {delta}, threshold: {filter_threshold}, should delete: {delta >= filter_threshold}"
                    )

                    if delta >= filter_threshold:
                        active_instruction_candidates_indices[predict_index].remove(p_prime_index)
                        inactive_instruction_candidates_indices[predict_index].add(p_prime_index)

                        n_p_mask[predict_index][p_prime_index] += budget
                        u_p_mask[predict_index][p_prime_index] += 1e9

                pbar.update(1)

        winner_candidate_indices = np.argmax(u_p - u_p_mask, axis=1)

        print(
            f"Finished the process of continous rejecting, remaining candidate for each predict: {active_instruction_candidates_indices}"
        )

        # Set the instruction to be the last remaining instruction
        for predict_index, predict in enumerate(program.predictors()):
            print(
                f"Winning instruction for predict {predict_index}: {instruction_candidates[predict_index][winner_candidate_indices[predict_index]]}"
            )
            selected_instruction = instruction_candidates[predict_index][winner_candidate_indices[predict_index]]
            updated_signature = get_signature(predict).with_instructions(selected_instruction)
            set_signature(predict, updated_signature)

        return program

    def _triple_csar(self, program, demo_candidates, trainset, valset=None, budget=None, total_demos=4):
        num_predicts = len(demo_candidates)
        num_candidates_per_predict = len(demo_candidates[0])

        g_accept = [[] for _ in range(num_predicts)]
        g_reject = [[] for _ in range(num_predicts)]
        log_g = math.log(num_candidates_per_predict)

        active_demo_candidates = demo_candidates

        t_p_last = 0
        for step in range(num_candidates_per_predict):
            active_num_candidates_per_predict = len(active_demo_candidates[0])

            t_p = math.ceil((budget - num_candidates_per_predict) / (log_g * (num_candidates_per_predict - step)))

            num_trials_per_candidate = max(t_p - t_p_last, 1)
            t_p_last = t_p

            print(f"Number of trials for each candidate in step {step}: {num_trials_per_candidate}")

            # Make an array [0, 0, 0, ..., 1, 1, 1, ...], the repetition number is num_trials_per_candidate.
            demo_indices = [
                i for _ in range(num_trials_per_candidate) for i in range(active_num_candidates_per_predict)
            ]

            demo_indices_sequence_all = []
            for j in range(num_predicts):
                # Generate an demo index sequence for each predict.
                demo_indices_sequence = list(demo_indices)
                random.shuffle(demo_indices_sequence)
                demo_indices_sequence_all.append(demo_indices_sequence)

            # Transpose so that each element is a sequence of index of the
            # instruction candidate for each predict. E.g, if there are 2
            # predicts and each has 3 candidates, after transpose
            # `instruction_indices_sequence_all` looks like
            # [[0, 1, 1], [2, 1, 0], ...]
            demo_indices_sequence_all = np.array(demo_indices_sequence_all).transpose().tolist()

            # The score book is of shape Np * Ni * Nt
            # Np: number of predicts in the DSPy program
            # Ni: number of instructions candidates for each predict
            # Nt: number of trials required for each instruction candidate
            score_book = np.empty(
                (
                    len(active_demo_candidates),
                    len(active_demo_candidates[0]),
                    0,
                )
            ).tolist()

            with tqdm(total=len(demo_indices_sequence_all)) as pbar:
                for demo_indices in demo_indices_sequence_all:
                    # Execute with random demo for each module
                    score = self._execute_one_demo_combo(
                        program,
                        active_demo_candidates,
                        demo_indices,
                        valset,
                    )

                    # Update score_book with the result
                    for predict_index, demo_index in enumerate(demo_indices):
                        score_book[predict_index][demo_index].append(score)

                    # Manually update the progress bar
                    pbar.update(1)

            # Calculate the average score for each predict-demo pair.
            score_book = np.array(score_book)
            mean_score = np.mean(score_book, axis=-1)

            print(f"Score book in round {step}")
            print(mean_score)

            for predict_index in range(num_predicts):
                if len(g_accept[predict_index]) >= total_demos:
                    # If for this predict, we have already gotten enough demos, skip.
                    continue
                sorted_value_and_index = sorted(enumerate(mean_score[predict_index]), key=lambda x: x[1], reverse=True)

                highest_anchor_gap = (
                    sorted_value_and_index[0][1] - sorted_value_and_index[total_demos - len(g_accept[0])][1]
                )
                lowest_anchor_gap = (
                    sorted_value_and_index[total_demos - len(g_accept[0]) - 1][1] - sorted_value_and_index[-1][1]
                )

                if highest_anchor_gap >= lowest_anchor_gap:
                    demo_index = sorted_value_and_index[0][0]
                    g_accept[predict_index].append(active_demo_candidates[predict_index][demo_index])
                    del active_demo_candidates[predict_index][demo_index]
                else:
                    demo_index = sorted_value_and_index[-1][0]
                    g_reject[predict_index].append(active_demo_candidates[predict_index][demo_index])
                    del active_demo_candidates[predict_index][demo_index]

        print(f"Searching finished, winning candidate: {g_accept}")

        for predict_index, predict in enumerate(program.predictors()):
            predict.demos.extend(g_accept[predict_index])

        return program

    def compile(self, program, trainset, valset=None, budget=None, total_demos=5):
        program = program.deepcopy()
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset)

        if self.algorithm == "SH":
            instruction_candidates = self._propose_instructions(
                program,
                trainset,
                demo_candidates=demo_candidates,
            )

            return self._triple_sh(
                program,
                instruction_candidates,
                trainset,
                valset=valset,
                budget=budget,
            )

        if self.algorithm == "CR":
            instruction_candidates = self._propose_instructions(
                program,
                trainset,
                demo_candidates=demo_candidates,
            )

            return self._triple_cr(
                program,
                instruction_candidates,
                trainset,
                valset=valset,
                budget=budget,
            )

        if self.algorithm == "CSAR":
            demo_candidates_2d = []
            for predict_index in range(len(program.predictors())):
                demo_candidates_2d.append(demo_candidates[predict_index][1])

            return self._triple_csar(
                program,
                demo_candidates_2d,
                trainset,
                valset=valset,
                budget=budget,
                total_demos=total_demos,
            )
