import inspect
import logging
import math
import os
import random
import shutil
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from juptyer notebooks
    extract_symbols = None

import dspy
from dspy.teleprompt.bootstrap import BootstrapFewShot, LabeledFewShot

"""
This file consists of helper functions for our variety of optimizers.
"""

### OPTIMIZER TRAINING UTILS ###


def create_minibatch(trainset, batch_size=50, rng=None):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # If no RNG is provided, fall back to the global random instance
    rng = rng or random

    # Randomly sample indices for the mini-batch using the provided rng
    sampled_indices = rng.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch


def eval_candidate_program(batch_size, trainset, candidate_program, evaluate, rng=None):
    """Evaluate a candidate program on the trainset, using the specified batch size."""

    try:
        # Evaluate on the full trainset
        if batch_size >= len(trainset):
            score = evaluate(candidate_program, devset=trainset)
        # Or evaluate on a minibatch
        else:
            score = evaluate(
                candidate_program,
                devset=create_minibatch(trainset, batch_size, rng),
            )
    except Exception as e:
        print(f"Exception occurred: {e}")
        score = 0.0  # TODO: Handle this better, as -ve scores are possible

    return score


def eval_candidate_program_with_pruning(
    trial, trial_logs, trainset, candidate_program, evaluate, trial_num, batch_size=100,
):
    """Evaluation of candidate_program with pruning implemented"""

    # Evaluate with the new prompts
    total_score = 0
    num_batches = math.ceil(len(trainset) / batch_size)
    total_eval_size = 0

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(trainset))
        split_trainset = trainset[start_index:end_index]
        split_score = evaluate(
            candidate_program, devset=split_trainset, display_table=0,
        )
        print(f"{i}st split score: {split_score}")
        total_eval_size += len(split_trainset)

        total_score += split_score * len(split_trainset)
        curr_weighted_avg_score = total_score / min((i + 1) * batch_size, len(trainset))
        print(f"curr average score: {curr_weighted_avg_score}")

        trial.report(curr_weighted_avg_score, i)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("Trial pruned.")
            trial_logs[trial_num]["score"] = curr_weighted_avg_score
            trial_logs[trial_num]["num_eval_calls"] = total_eval_size
            trial_logs[trial_num]["pruned"] = True
            return curr_weighted_avg_score, trial_logs, total_eval_size, True

    print(f"Fully evaled score: {curr_weighted_avg_score}")
    score = curr_weighted_avg_score

    trial_logs[trial_num]["full_eval"] = False
    trial_logs[trial_num]["score"] = score
    trial_logs[trial_num]["pruned"] = False
    return score, trial_logs, total_eval_size, False


def get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos):
    """Used as a helper function for bayesian + minibatching optimizers. Returns the program with the highest average score from the batches evaluated so far."""

    # Calculate the mean for each combination of categorical parameters, based on past trials
    results = []
    for key, values in param_score_dict.items():
        scores = np.array([v[0] for v in values])
        mean = np.average(scores)
        program = values[0][1]
        results.append((key, mean, program))

    # Sort results by the mean
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find the combination with the highest mean, skip fully evaluated ones
    for combination in sorted_results:
        key, mean, program = combination

        if key in fully_evaled_param_combos:
            continue

        return program, mean, key

    # If no valid program is found, we return the last valid one that we found
    return program, mean, key


def calculate_last_n_proposed_quality(
    base_program, trial_logs, evaluate, trainset, devset, n,
):
    """
    Calculate the average and best quality of the last n programs proposed. This is useful for seeing if our proposals
    are actually 'improving' overtime or not.
    """
    # Get the trials from the last n keys in trial logs
    last_n_trial_nums = list(trial_logs.keys())[-n:]

    # Calculate the average and best score of these trials
    # if num_eval_calls in the trial is less than the trainset, throw a not-implemented error for now
    total_train_score = 0
    best_train_score = 0
    total_dev_score = 0
    best_dev_score = 0
    for trial_num in last_n_trial_nums:
        full_eval = trial_logs[trial_num]["full_eval"]
        if not full_eval:
            raise NotImplementedError(
                "Still need to implement non full eval handling in calculate_last_n_proposed_quality",
            )
        train_score = trial_logs[trial_num]["score"]
        program = base_program.deepcopy()
        program.load(trial_logs[trial_num]["program_path"])

        dev_score = evaluate(program, devset=devset)

        total_train_score += train_score
        total_dev_score += dev_score
        if train_score > best_train_score:
            best_train_score = train_score
            best_dev_score = dev_score

    return best_train_score, total_train_score / n, best_dev_score, total_dev_score / n


### LOGGING UTILS ###


def get_task_model_history_for_full_example(
    candidate_program, task_model, devset, evaluate,
):
    """Get a full trace of the task model's history for a given candidate program."""
    _ = evaluate(candidate_program, devset=devset[:1])
    _ = task_model.inspect_history(n=len(candidate_program.predictors()))
    return task_model.inspect_history(n=len(candidate_program.predictors()))


def print_full_program(program):
    """Print out the program's instructions & prefixes for each module."""
    for i, predictor in enumerate(program.predictors()):
        print(f"Predictor {i}")
        print(f"i: {get_signature(predictor).instructions}")
        *_, last_field = get_signature(predictor).fields.values()
        print(f"p: {last_field.json_schema_extra['prefix']}")
    print("\n")


def save_candidate_program(program, log_dir, trial_num, note=None):
    """Save the candidate program to the log directory."""

    if log_dir is None:
        return None

    # Ensure the directory exists
    eval_programs_dir = os.path.join(log_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)

    # Define the save path for the program
    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}")

    # Save the program
    program.save(save_path)

    return save_path


def save_file_to_log_dir(source_file_path, log_dir):
    if log_dir is None:
        return
    """Save a file to our log directory"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    destination_file_path = os.path.join(log_dir, os.path.basename(source_file_path))

    # Copy the file
    shutil.copy(source_file_path, destination_file_path)


def setup_logging(log_dir):
    """Setup logger, which will log our print statements to a txt file at our log_dir for later viewing"""
    if log_dir is None:
        return
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Create a file handler that logs debug and higher level messages
    file_handler = logging.FileHandler(f"{log_dir}/logs.txt")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


### OTHER UTILS ###

def get_prompt_model(prompt_model):
    if prompt_model:
        return prompt_model
    else:
        return dspy.settings.lm

def get_signature(predictor):
    if hasattr(predictor, "extended_signature"):
        return predictor.extended_signature
    elif hasattr(predictor, "signature"):
        return predictor.signature
    return None


def set_signature(predictor, updated_signature):
    if hasattr(predictor, "extended_signature"):
        predictor.extended_signature = updated_signature
    elif hasattr(predictor, "signature"):
        predictor.signature = updated_signature


def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets,
    trainset,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    teacher_settings,
    max_errors=10,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
    parallel_bootstrapping=False,
):
    """
    This function creates few-shot examples in the same way that random search does.
    If parallel_bootstrapping is True, it parallelizes the bootstrapping process over the seeds.
    """
    demo_candidates = {}

    # Account for the way this is set up, where we add in 3 more candidate sets to the N specified
    total_candidate_sets = num_candidate_sets + 3

    # Initialize demo_candidates dictionary
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i] = []

    # Prepare the list of seeds
    seeds = list(range(-3, num_candidate_sets))

    # Define the function to create a candidate set for a given seed
    def create_candidate(seed):
        trainset_copy = list(trainset)  # Copy of trainset

        print(f"Bootstrapping set {seed + 4}/{total_candidate_sets}")

        if seed == -3 and include_non_bootstrapped:
            # Zero-shot
            program2 = student.reset_copy()

        elif (
            seed == -2
            and max_labeled_demos > 0
            and include_non_bootstrapped
        ):
            # Labels only
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = teleprompter.compile(
                student, trainset=trainset_copy, sample=labeled_sample,
            )

        elif seed == -1:
            # Unshuffled few-shot
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = program.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            # Shuffled few-shot
            random.Random(seed).shuffle(trainset_copy)
            size = random.Random(seed).randint(min_num_samples, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )

            program2 = teleprompter.compile(
                student, teacher=teacher, trainset=trainset_copy,
            )

        # Collect the demos for each predictor
        predictor_demos = {}
        for i, _ in enumerate(student.predictors()):
            predictor_demos[i] = program2.predictors()[i].demos

        return seed, predictor_demos

    if parallel_bootstrapping:
        # Use ThreadPoolExecutor to parallelize the creation of candidate sets
        num_threads = total_candidate_sets  # As per your request

        # Dictionary to store the results in order
        seed_to_result = {}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_seed = {executor.submit(create_candidate, s): s for s in seeds}
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    seed_result, predictor_demos = future.result()
                    seed_to_result[seed] = predictor_demos
                except Exception as e:
                    print(f"Error creating candidate set for seed {seed}: {e}")
                    # Handle exceptions as needed

        # After all threads have completed, collect the results in order
        for seed in sorted(seed_to_result.keys()):
            predictor_demos = seed_to_result[seed]
            for i in demo_candidates:
                demo_candidates[i].append(predictor_demos[i])

    else:
        # Sequential execution
        for seed in seeds:
            seed_result, predictor_demos = create_candidate(seed)
            # Append the demos to demo_candidates
            for i in demo_candidates:
                demo_candidates[i].append(predictor_demos[i])

    return demo_candidates

def old_getfile(object):
    """Work out which source or compiled file an object was defined in."""
    if inspect.ismodule(object):
        if getattr(object, '__file__', None):
            return object.__file__
        raise TypeError('{!r} is a built-in module'.format(object))
    if inspect.isclass(object):
        if hasattr(object, '__module__'):
            module = sys.modules.get(object.__module__)
            if getattr(module, '__file__', None):
                return module.__file__
            if object.__module__ == '__main__':
                raise OSError('source code not available')
        raise TypeError('{!r} is a built-in class'.format(object))
    if inspect.ismethod(object):
        object = object.__func__
    if inspect.isfunction(object):
        object = object.__code__
    if inspect.istraceback(object):
        object = object.tb_frame
    if inspect.isframe(object):
        object = object.f_code
    if inspect.iscode(object):
        return object.co_filename
    raise TypeError('module, class, method, function, traceback, frame, or '
                    'code object was expected, got {}'.format(
                    type(object).__name__))

def new_getfile(object):
    if not inspect.isclass(object):
        return old_getfile(object)
    
    # Lookup by parent module (as in current inspect)
    if hasattr(object, '__module__'):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, '__file__'):
            return object_.__file__
    
    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + '.' + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    raise TypeError(f'Source for {object!r} not found')

inspect.getfile = new_getfile