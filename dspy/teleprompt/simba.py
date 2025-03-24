import dspy
import random
import logging

import numpy as np
from typing import Callable
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.simba_utils import prepare_models_for_resampling, wrap_program, append_a_demo, append_a_rule

logger = logging.getLogger(__name__)


# Stochastic Introspective Mini-Batch Ascent
class SIMBA(Teleprompter):
    def __init__(
        self,
        *,
        metric: Callable,
        bsize=32,
        num_candidates=6,
        max_steps=8,
        max_demos=4,
        demo_input_field_maxlen=100_000,
        num_threads=16,
        temperature_for_sampling=0.2,
        temperature_for_candidates=0.2,
    ):
        """
        :param metric: A function (Example, prediction_dict) -> float
        :param bsize: mini-batch size
        :param num_candidates: how many new candidate programs to produce per iteration
        :param max_steps: how many optimization steps to run
        :param max_demos: how many demos we allow a predictor to hold before we must drop some
        :param demo_input_field_maxlen: how many characters of an input field to keep when building a new demo
        :param num_threads: how many threads for run_parallel
        :param temperature_for_sampling: temperature used for picking programs for the trajectory-sampling step
        :param temperature_for_candidates: temperature used for picking the source program for building new candidates
        """
        self.metric = metric
        self.bsize = bsize
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.demo_input_field_maxlen = demo_input_field_maxlen
        self.num_threads = num_threads

        self.temperature_for_sampling = temperature_for_sampling
        self.temperature_for_candidates = temperature_for_candidates

        self.strategies = [append_a_demo(demo_input_field_maxlen), append_a_rule]

    def compile(self, student: dspy.Module, *, trainset: list[dspy.Example], seed: int = 0):
        # Basic checks
        assert len(trainset) >= self.bsize, f"Trainset too small: {len(trainset)} < {self.bsize}"

        # Initialize RNG
        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        programs = []
        program_scores = {}
        next_program_idx = 0

        # Helper functions
        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        def top_k_plus_baseline(k: int) -> list[int]:
            # Sort all programs by descending average score
            scored_programs = sorted(programs, key=lambda p: calc_average_score(p.simba_idx), reverse=True)
            top_k = [p.simba_idx for p in scored_programs[:k]]
            # Ensure baseline=0 is in there:
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def softmax_sample(rng_obj: random.Random, program_idxs: list[int], temperature: float) -> int:
            if not program_idxs:
                raise ValueError("No programs available for softmax sampling.")

            # Unnormalized weights
            scores = [calc_average_score(idx) for idx in program_idxs]
            exps = [np.exp(s / temperature) for s in scores]
            sum_exps = sum(exps)
            if sum_exps <= 0:
                # Fallback: uniform if all exps are zero
                return rng_obj.choice(program_idxs)

            # Weighted random choice
            probs = [val / sum_exps for val in exps]
            return rng_obj.choices(program_idxs, weights=probs, k=1)[0]

        def register_new_program(prog: dspy.Module, score_list: list[float]):
            nonlocal next_program_idx
            next_program_idx += 1
            new_idx = next_program_idx
            prog.simba_idx = new_idx
            programs.append(prog)
            program_scores[new_idx] = score_list

        # Initialize the baseline program: index=0
        student = student.deepcopy()
        student.simba_idx = 0
        programs.append(student)
        program_scores[0] = []

        winning_programs = [student]

        # Data shuffling
        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0

        # Parallel runner
        run_parallel = dspy.Parallel(access_examples=False, num_threads=self.num_threads)

        for batch_idx in range(self.max_steps):
            logger.info(f"Starting batch {batch_idx+1} of {self.max_steps}.")

            # STEP 1: Get next batch
            if instance_idx + self.bsize > len(trainset):
                rng.shuffle(data_indices)
                instance_idx = 0

            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize

            # We'll generate (program, model) pairs for the trajectory sampling.
            # Prepare distinct LMs (with different temperatures, etc.) from the baseline=programs[0].
            models = prepare_models_for_resampling(programs[0], self.num_candidates)
            top_programs = top_k_plus_baseline(self.num_candidates)

            exec_pairs = []
            predictor2name = {}

            # For each model, for each example, pick a program from the pool via softmax
            for model in models:
                for example in batch:
                    chosen_prog_idx = softmax_sample(rng, top_programs, self.temperature_for_sampling)
                    candidate_system = programs[chosen_prog_idx].deepcopy()
                    candidate_system.set_lm(model)

                    for name, predictor in candidate_system.named_predictors():
                        predictor2name[id(predictor)] = name

                    # Use the special wrap that includes the 'example' in the output
                    wrapped_candidate_system = wrap_program(candidate_system, self.metric)
                    exec_pairs.append((wrapped_candidate_system, example))

            # STEP 2: Execute
            logger.info(f"Sampling program trajectories on {self.bsize} examples x {self.num_candidates} samples.")
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(exec_pairs) == self.bsize * self.num_candidates

            # STEP 3: Sort the training buckets by (max-to-min gap, max score, and max-to-avg gap).
            buckets = []
            largest_max_to_avg_gap = float("-inf")
            batch_10th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 10)
            batch_90th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 90)

            # We'll chunk `outputs` by example index, each chunk has length = num_candidates
            for idx, example in enumerate(batch):
                # gather all results for this example
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)

                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                max_to_min_gap = max_score - min_score
                max_to_avg_gap = max_score - avg_score
                if max_to_avg_gap > largest_max_to_avg_gap:
                    largest_max_to_avg_gap = max_to_avg_gap

                buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))

            # sort the buckets
            buckets.sort(key=lambda x: x[1], reverse=True)

            # Baseline for the batch is just the average of all runs
            all_scores_in_this_batch = [o["score"] for o in outputs]
            baseline_score = sum(all_scores_in_this_batch) / len(all_scores_in_this_batch)
            logger.info(f"Batch {batch_idx+1}: Baseline mini-batch score: {baseline_score}\n")

            # STEP 4: Build new candidate programs by applying a strategy to some top buckets.
            system_candidates = []
            for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
                max_to_min_gap, max_score, max_to_avg_gap = bucket_stats
                logger.info(
                    f"Batch {batch_idx+1}: Processing bucket #{bucket_idx+1}, with max score {max_score}, "
                    f"max-to-min gap {max_to_min_gap}, and max-to-avg gap {max_to_avg_gap}."
                )

                # pick source program
                src_prog_idx = softmax_sample(
                    rng, top_k_plus_baseline(self.num_candidates), self.temperature_for_candidates
                )
                system_candidate = programs[src_prog_idx].deepcopy()

                # Drop some demos from each predictor
                name2predictor = {}
                num_demos_list = []

                for name, predictor in system_candidate.named_predictors():
                    name2predictor[name] = predictor
                    num_demos_list.append(len(predictor.demos))

                num_demos = max(num_demos_list) if num_demos_list else 0
                num_demos_to_drop = max(rng_np.poisson(num_demos / self.max_demos), int(num_demos >= self.max_demos))
                num_demos_to_drop = min(num_demos_to_drop, num_demos)
                demos_to_drop = [rng.randrange(num_demos) for _ in range(num_demos_to_drop)]

                for name, predictor in name2predictor.items():
                    predictor.demos = [demo for idxd, demo in enumerate(predictor.demos) if idxd not in demos_to_drop]

                # Pick a strategy
                strategy = rng.choice(self.strategies)
                logger.info(
                    f"Batch {batch_idx+1}: Invoking strategy: {strategy.__name__}"
                    + (f", having dropped {num_demos_to_drop} demos per predictor" if num_demos_to_drop else "")
                )

                try:
                    strategy(
                        bucket,
                        system_candidate,
                        predictor2name=predictor2name,
                        name2predictor=name2predictor,
                        batch_10p_score=batch_10th_percentile_score,
                        batch_90p_score=batch_90th_percentile_score,
                    )
                except Exception as e:
                    logger.error(f"Strategy failed with error: {e}")
                    continue

                system_candidates.append(system_candidate)
                logger.info("\n")

                if len(system_candidates) >= self.num_candidates + 1:
                    break

            # STEP 5: Evaluate these new system_candidates on the same mini-batch
            logger.info(f"Batch {batch_idx+1}: Evaluating {len(system_candidates)} programs on {self.bsize} examples.")

            exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in system_candidates for ex in batch]
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(exec_pairs) == len(system_candidates) * self.bsize

            # STEP 6: Compute average mini-batch scores for each new candidate
            candidate_scores = []
            for idx_cand, cand_sys in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                avg_sys_score = sum(sys_scores) / len(sys_scores)
                candidate_scores.append(avg_sys_score)

            logger.info(
                f"Scores after {batch_idx+1} batches: {candidate_scores}, "
                f"Best: {max(candidate_scores) if candidate_scores else 'N/A'}\n"
            )

            # STEP 7: Select the best among these new ones for "winning" record
            if candidate_scores:
                best_idx_among_candidates = candidate_scores.index(max(candidate_scores))
                best_program = system_candidates[best_idx_among_candidates]
                winning_programs.append(best_program)

            # STEP 8: Register all new candidate systems in our global pool
            for idx_cand, cand_sys in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                register_new_program(cand_sys, sys_scores)

        M = len(winning_programs) - 1
        N = self.num_candidates + 1
        if M < 1:
            # Only one or zero winning programs
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]
        program_idxs = list(dict.fromkeys(program_idxs))

        candidate_programs = [winning_programs[i] for i in program_idxs]
        logger.info(f"VALIDATION: Evaluating {len(candidate_programs)} programs on the full trainset.")
        exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in candidate_programs for ex in trainset]
        outputs = run_parallel(exec_pairs)

        scores = []
        for idx_prog, prog in enumerate(candidate_programs):
            start = idx_prog * len(trainset)
            end = (idx_prog + 1) * len(trainset)
            sys_scores = [outputs[i]["score"] for i in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)

        best_idx = scores.index(max(scores)) if scores else 0
        best_program = candidate_programs[best_idx]
        logger.info(
            f"Final trainset scores: {scores}, Best: {max(scores) if scores else 'N/A'} "
            f"(at index {best_idx if scores else 'N/A'})\n\n\n"
        )

        # FIXME: Attach all program candidates in decreasing average score to the best program.
        best_program.candidate_programs = candidate_programs
        best_program.winning_programs = winning_programs

        return best_program
