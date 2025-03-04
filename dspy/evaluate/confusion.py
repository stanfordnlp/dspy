import contextlib
import random
import signal
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from re import findall
from typing import Dict, List, Tuple

import numpy as np
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import dspy
from dspy import LabeledFewShot, BootstrapFewShot
from dspy.evaluate.evaluate import *
from dspy.teleprompt.teleprompt import Teleprompter


def most_votes(votes, tiebreaker=None):
    """
    Determine the most common element in a list with optional tiebreaking methods.

    Args:
        votes (List): A list of votes/labels.
        tiebreaker (str, optional): Method to handle ties. Options: "first", "last".

    Returns:
        The most common element or, in case of ties, the one selected by the tiebreaker method.
    """
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    if len(winners) == 1:
        return winners[0]
    elif tiebreaker == "first":
        return min(winners, key=votes.index)
    elif tiebreaker == "last":
        return max(winners, key=votes.index)


class Confusion:
    """
    Evaluate DSPy programs using Matthews Correlation Coefficient (MCC) as the metric.

    This class is used to evaluate classification programs by comparing predictions
    against expected labels and computing a confusion matrix and Matthews correlation coefficient.

    Attributes:
        devset (List[dspy.Example]): The evaluation dataset.
        num_threads (int): Number of threads for parallel evaluation.
        display_progress (bool): Whether to display progress during evaluation.
        display_table (bool): Whether to display results in a table.
        max_errors (int): Maximum number of errors before stopping evaluation.
        return_matrix (bool): Whether to return the confusion matrix.
        return_outputs (bool): Whether to return outputs for each example.
        provide_traceback (bool): Whether to provide traceback information.
        use_class_weight (bool): Whether to use class weighting in computing metrics.
        output_field (str): The field in predictions to use as the output label.
        match (callable, optional): Function to match extracted values to labels.
    """

    def __init__(
            self,
            *,
            devset,
            num_threads=1,
            display_progress=False,
            display_table=False,
            max_errors=5,
            return_matrix=False,
            return_outputs=False,
            provide_traceback=False,
            use_class_weight=True,
            output_field="response",
            match="last",
            **_kwargs,
    ):
        """
        Initialize the Confusion evaluator.

        Args:
            devset (List[dspy.Example]): The evaluation dataset.
            num_threads (int): Number of threads for parallel evaluation.
            display_progress (bool): Whether to display progress during evaluation.
            display_table (bool): Whether to display results in a table.
            max_errors (int): Maximum number of errors before stopping evaluation.
            return_matrix (bool): Whether to return the confusion matrix.
            return_outputs (bool): Whether to return outputs for each example.
            provide_traceback (bool): Whether to provide traceback information.
            use_class_weight (bool): Whether to use class weighting in computing metrics.
            output_field (str): The field in predictions to use as the output label.
            match (callable, str, optional): Function or method to match extracted values to labels.
                                           Options: "first", "last", "most", or a custom function.
        """
        self.devset = devset
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.return_matrix = return_matrix
        self.return_outputs = return_outputs
        self.provide_traceback = provide_traceback
        self.use_class_weight = use_class_weight
        self.output_field = output_field
        if callable(match):
            self.match = match
        elif match == "first":
            self.match = lambda x: x[0]
        elif match == "last":
            self.match = lambda x: x[-1]
        elif match == "most":
            self.match = most_votes
        else:
            raise ValueError(f"Invalid match function: {match}")

    def _extract(self, response, labels):
        """
        Extract labels from a response by finding matches.

        Args:
            response (str): The response text to search in.
            labels (List[str]): The list of labels to look for.

        Returns:
            The matched label according to the match function.
        """
        found = findall(r"|".join(labels), response.lower())
        if found:
            return self.match(found)
        else:
            return response

    def construct_labels_and_matrix(self, devset, preds=None):
        """
        Construct the list of unique labels and optionally the confusion matrix.

        Args:
            devset (List[Tuple]): List of (index, example) tuples.
            preds (Dict[str, List[str]], optional): Predictions by label.

        Returns:
            If preds is None, returns the list of unique labels.
            If preds is provided, returns a tuple (labels, confusion_matrix).
        """
        classes = [arg[self.output_field].lower() for _, arg in devset]
        labels = np.unique(classes).tolist()

        if preds is None:
            return labels

        # Calculate class weights
        if self.use_class_weight:
            class_counts = Counter(classes)
            weight = {k: 1 / v for k, v in class_counts.items()}
        else:
            weight = {k: 1 for k in labels}

        # Initialize the confusion matrix
        confusion_matrix = np.zeros([len(labels)] * 2, dtype=np.float64)

        # Get model answers
        answers = {label: [self._extract(pred, labels) for pred in preds[label]] for label in labels}

        # Fill the confusion matrix
        for idx, label in enumerate(labels):
            for answer in answers[label]:
                if answer in labels:
                    confusion_matrix[idx][labels.index(answer)] += weight[label]

        return labels, confusion_matrix

    def get_matthews_corrcoef(self, devset, preds, return_cm=False):
        """
        Calculate Matthews Correlation Coefficient from predictions.

        Args:
            devset (List[Tuple]): List of (index, example) tuples.
            preds (Dict[str, List[str]]): Predictions by label.
            return_cm (bool): Whether to return the confusion matrix.

        Returns:
            If return_cm is False, returns the MCC score.
            If return_cm is True, returns a tuple (mcc_score, confusion_matrix_dataframe).
        """
        labels, C = self.construct_labels_and_matrix(devset, preds)

        # <sklearn.metrics.matthews_corrcoef>
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_samples = p_sum.sum()
        n_samples_2 = n_samples ** 2
        cov_ypyp = n_samples_2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples_2 - np.dot(t_sum, t_sum)
        prod = cov_ypyp * cov_ytyt

        if prod == 0:
            out = 0.0
        else:
            n_correct = np.trace(C, dtype=np.float64)
            cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
            out = cov_ytyp / np.sqrt(prod)
        # </sklearn.metrics.matthews_corrcoef>

        if return_cm:
            cm = pd.DataFrame(C,
                              index=pd.Index(labels, name="Actual"),
                              columns=pd.Index(labels, name="Predicted"))
            return out, cm
        return out

    def _update_progress(self, pbar, preds, devset):
        """
        Update the progress bar with the current MCC score.

        Args:
            pbar (tqdm.tqdm): The progress bar object.
            preds (Dict[str, List[str]]): Predictions by label.
            devset (List[Tuple]): List of (index, example) tuples.
        """
        mcc = self.get_matthews_corrcoef(devset, preds)
        pbar.set_description(f"MCC: {mcc:.6f}")
        pbar.update()

    def _execute_single_thread(self, wrapped_program, devset, display_progress, preds):
        """
        Execute the program on the devset using a single thread.

        Args:
            wrapped_program (callable): The program wrapper function.
            devset (List[Tuple]): List of (index, example) tuples.
            display_progress (bool): Whether to display progress.
            preds (Dict[str, List]): Dictionary to collect predictions by label.

        Returns:
            List[Tuple]: List of (example_idx, example, prediction) tuples.
        """
        reordered_devset = []

        pbar = tqdm.tqdm(total=len(devset), dynamic_ncols=True, disable=not display_progress, file=sys.stdout)
        for idx, arg in devset:
            with logging_redirect_tqdm():
                example_idx, example, prediction = wrapped_program(idx, arg)
                reordered_devset.append((example_idx, example, prediction))
                preds[arg[self.output_field].lower()].append(prediction.get(self.output_field, "error"))
                self._update_progress(pbar, preds, devset)

        pbar.close()

        return reordered_devset

    def _execute_multi_thread(self, wrapped_program, devset, num_threads, display_progress, preds):
        """
        Execute the program on the devset using multiple threads.

        Args:
            wrapped_program (callable): The program wrapper function.
            devset (List[Tuple]): List of (index, example) tuples.
            num_threads (int): Number of threads to use.
            display_progress (bool): Whether to display progress.
            preds (Dict[str, List]): Dictionary to collect predictions by label.

        Returns:
            List[Tuple]: Reordered list of (example_idx, example, prediction) tuples.
        """
        reordered_devset = []
        job_cancelled = "cancelled"

        # context manager to handle sigint
        @contextlib.contextmanager
        def interrupt_handler_manager():
            """Sets the cancel_jobs event when a SIGINT is received."""
            default_handler = signal.getsignal(signal.SIGINT)

            def interrupt_handler(sig, frame):
                self.cancel_jobs.set()
                logger.warning("Received SIGINT. Cancelling evaluation.")
                default_handler(sig, frame)

            signal.signal(signal.SIGINT, interrupt_handler)
            yield
            # reset to the default handler
            signal.signal(signal.SIGINT, default_handler)

        def cancellable_wrapped_program(idx, arg):
            # If the cancel_jobs event is set, return the cancelled_job literal
            if self.cancel_jobs.is_set():
                return None, None, job_cancelled, None
            return arg, wrapped_program(idx, arg)

        with ThreadPoolExecutor(max_workers=num_threads) as executor, interrupt_handler_manager():
            futures = {executor.submit(cancellable_wrapped_program, idx, arg) for idx, arg in devset}
            pbar = tqdm.tqdm(total=len(devset), dynamic_ncols=True, disable=not display_progress)

            for future in as_completed(futures):
                arg, (example_idx, example, prediction) = future.result()

                # use the cancelled_job literal to check if the job was cancelled - use "is" not "=="
                # in case the prediction is "cancelled" for some reason.
                if prediction is job_cancelled:
                    continue

                reordered_devset.append((example_idx, example, prediction))
                preds[arg[self.output_field].lower()].append(prediction.get(self.output_field, "error"))
                self._update_progress(pbar, preds, devset)
            pbar.close()

        if self.cancel_jobs.is_set():
            logger.warning("Evaluation was cancelled. The results may be incomplete.")
            raise KeyboardInterrupt

        return reordered_devset

    def __call__(
            self,
            program,
            devset=None,
            num_threads=None,
            display_progress=None,
            display_table=None,
            return_matrix=None,
            return_outputs=None,
    ):
        """
        Evaluate a DSPy program on a dataset using MCC.

        Args:
            program (dspy.Module): The DSPy program to evaluate.
            devset (List[dspy.Example], optional): The evaluation dataset.
            num_threads (int, optional): Number of threads for parallel evaluation.
            display_progress (bool, optional): Whether to display progress.
            display_table (bool, optional): Whether to display a table of results.
            return_matrix (bool, optional): Whether to return the confusion matrix.
            return_outputs (bool, optional): Whether to return outputs for each example.

        Returns:
            Depending on the flags, returns combinations of:
            - float: The MCC score
            - List[Tuple]: Results for each example
            - pandas.DataFrame: The confusion matrix
        """
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_matrix = return_matrix if return_matrix is not None else self.return_matrix
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        results = []

        def wrapped_program(example_idx, example):
            try:
                prediction = program(**example.inputs())

                # increment assert and suggest failures to program's attributes
                if hasattr(program, "_assert_failures"):
                    program._assert_failures += dspy.settings.get("assert_failures")
                if hasattr(program, "_suggest_failures"):
                    program._suggest_failures += dspy.settings.get("suggest_failures")

                return example_idx, example, prediction
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count
                if current_error_count >= self.max_errors:
                    raise e

                if self.provide_traceback:
                    logger.error(
                        f"Error for example in dev set: \t\t {e}\n\twith inputs:\n\t\t{example.inputs()}\n\nStack trace:\n\t{traceback.format_exc()}"
                    )
                else:
                    logger.error(
                        f"Error for example in dev set: \t\t {e}. Set `provide_traceback=True` to see the stack trace."
                    )

                return example_idx, example, {self.output_field: "error"}

        devset = list(enumerate(devset))
        tqdm.tqdm._instances.clear()

        labels = self.construct_labels_and_matrix(devset)

        preds = {label: [] for label in labels}

        if num_threads == 1:
            reordered_devset = self._execute_single_thread(wrapped_program, devset, display_progress, preds)
        else:
            reordered_devset = self._execute_multi_thread(
                wrapped_program,
                devset,
                num_threads,
                display_progress,
                preds,
            )

        mcc, cm = self.get_matthews_corrcoef(devset, preds, return_cm=True)

        logger.info(f"MCC: {mcc:.6f}")

        predicted_devset = sorted(reordered_devset)

        if return_outputs:  # Handle the return_outputs logic
            results = [(example, prediction) for _, example, prediction in predicted_devset]

        if display_table:
            data = [merge_dicts(example, prediction) for _, example, prediction in predicted_devset]

            result_df = pd.DataFrame(data)

            # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
            result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

            if isinstance(display_table, bool):
                df_to_display = result_df.copy()
                truncated_rows = 0
            else:
                df_to_display = result_df.head(display_table).copy()
                truncated_rows = len(result_df) - display_table

            display_dataframe(df_to_display)

            if truncated_rows > 0:
                # Simplified message about the truncated rows
                message = f"""
                <div style='
                    text-align: center;
                    font-size: 16px;
                    font-weight: bold;
                    color: #555;
                    margin: 10px 0;'>
                    ... {truncated_rows} more rows not displayed ...
                </div>
                """
                display(HTML(message))

        if return_matrix and return_outputs:
            return mcc, results, cm
        if return_matrix:
            return mcc, cm
        if return_outputs:
            return mcc, results

        return mcc


class MCCBootstrapFewShotWithRandomSearch(Teleprompter):
    """
    A Teleprompter that uses Matthews Correlation Coefficient for evaluating and
    bootstrapping few-shot examples with random search.

    This class compiles a program by bootstrapping demonstrations and evaluating
    candidates using MCC as the metric.
    """

    def __init__(
            self,
            teacher_settings={},
            max_bootstrapped_demos=4,
            max_labeled_demos=16,
            max_rounds=1,
            num_candidate_programs=16,
            num_threads=6,
            max_errors=10,
            stop_at_score=1.0,
            metric_threshold=None,
            use_class_weight=True,
            output_field="response",
    ):
        """
        Initialize the MCCBootstrapFewShotWithRandomSearch teleprompter.

        Args:
            teacher_settings (dict): Settings for the teacher.
            max_bootstrapped_demos (int): Maximum number of bootstrapped demos.
            max_labeled_demos (int): Maximum number of labeled demos.
            max_rounds (int): Maximum number of optimization rounds.
            num_candidate_programs (int): Number of candidate programs to evaluate.
            num_threads (int): Number of threads for parallel evaluation.
            max_errors (int): Maximum number of errors before stopping evaluation.
            stop_at_score (float, optional): Score at which to stop optimization.
            metric_threshold (float, optional): Threshold for the metric.
            use_class_weight (bool): Whether to use class weights.
            output_field (str): Field to use as the output.
        """
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        self.use_class_weight = use_class_weight
        self.output_field = output_field

        print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        """
        Compile a program by bootstrapping demonstrations and evaluating candidates.

        Args:
            student (dspy.Module): The student module to optimize.
            teacher (dspy.Module, optional): The teacher module.
            trainset (List[dspy.Example]): The training dataset.
            valset (List[dspy.Example], optional): The validation dataset.
            restrict (List[int], optional): Restrict optimization to these seeds.
            labeled_sample (bool): Whether to sample labeled examples.

        Returns:
            The best program found during optimization.
        """
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        scores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                continue

            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=self.max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                optimizer = BootstrapFewShot(
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=self.max_errors,
                )

                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            confusion = Confusion(
                devset=self.valset,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
                use_class_weight=self.use_class_weight,
                output_field=self.output_field,
            )

            score, cm = confusion(program, return_matrix=True)

            ############ Assertion-aware Optimization ############
            if hasattr(program, "_suggest_failures"):
                score = score - program._suggest_failures * 0.2
            if hasattr(program, "_assert_failures"):
                score = 0 if program._assert_failures > 0 else score
            ######################################################

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program

            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append((score,
                               cm,
                               seed,
                               program))

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(best_program.candidate_programs, key=lambda x: x[0], reverse=True)

        print(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program
