import random
import re
from collections import Counter

import numpy as np

from dspy import LabeledFewShot, BootstrapFewShot
from dspy.evaluate.evaluate import *
from dspy.teleprompt.teleprompt import Teleprompter


class Confusion:
    def __init__(
            self,
            *,
            labels,
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
            **_kwargs,
    ):
        self.labels = labels
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

    def extract(self, response):
        match = re.search(r"|".join(self.labels), response.lower())
        return match.group(0) if match else None

    def construct_matrix(self, preds, devset):
        labels = self.labels

        if self.use_class_weight:
            # use devset to get weights
            classes = [arg[self.output_field] for _, arg in devset]
            class_counts = Counter(classes)
            weight = {k: 1 / v for k, v in class_counts.items()}
        else:
            weight = {k: 1 for k in labels}

        # Initialize the confusion matrix
        confusion_matrix = np.zeros([len(labels)] * 2, dtype=np.float64)

        # Get model answers
        answers = {label: [self.extract(pred) for pred in preds[label]] for label in labels}

        # Fill the confusion matrix
        for idx, label in enumerate(labels):
            for answer in answers[label]:
                if answer in labels:
                    confusion_matrix[idx][labels.index(answer)] += weight[label]

        return confusion_matrix

    def get_matthews_corrcoef(self, preds, devset, return_cm=False):
        C = self.construct_matrix(preds, devset)
        # rest is from sklearn

        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

        if cov_ypyp * cov_ytyt == 0:
            out = 0.0
        else:
            out = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if return_cm:
            cm = pd.DataFrame(C,
                              index=pd.Index(self.labels, name="Actual"),
                              columns=pd.Index(self.labels, name="Predicted"))
            return out, cm
        return out

    def _execute_single_thread(self, wrapped_program, devset, display_progress, preds):
        reordered_devset = []

        pbar = tqdm.tqdm(total=len(devset), dynamic_ncols=True, disable=not display_progress, file=sys.stdout)
        for idx, arg in devset:
            with logging_redirect_tqdm():
                example_idx, example, prediction = wrapped_program(idx, arg)
                reordered_devset.append((example_idx, example, prediction))
                preds[arg[self.output_field]].append(prediction[self.output_field])
                self._update_progress(pbar, preds, devset)

        pbar.close()

        return reordered_devset

    def _execute_multi_thread(self, wrapped_program, devset, num_threads, display_progress, preds):
        reordered_devset = []
        job_cancelled = "cancelled"

        # context manger to handle sigint
        @contextlib.contextmanager
        def interrupt_handler_manager():
            """Sets the cancel_jobs event when a SIGINT is received."""
            default_handler = signal.getsignal(signal.SIGINT)

            def interrupt_handler(sig, frame):
                self.cancel_jobs.set()
                dspy.logger.warning("Received SIGINT. Cancelling evaluation.")
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
                preds[arg[self.output_field]].append(prediction[self.output_field])
                self._update_progress(pbar, preds, devset)
            pbar.close()

        if self.cancel_jobs.is_set():
            dspy.logger.warning("Evaluation was cancelled. The results may be incomplete.")
            raise KeyboardInterrupt

        return reordered_devset

    def _update_progress(self, pbar, preds, devset):
        mcc = self.get_matthews_corrcoef(preds, devset)
        pbar.set_description(f"MCC: {mcc:.6f}")
        pbar.update()

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
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_matrix = return_matrix if return_matrix is not None else self.return_matrix
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        results = []

        def wrapped_program(example_idx, example):
            # NOTE: TODO: Won't work if threads create threads!
            thread_stacks = dspy.settings.stack_by_thread
            creating_new_thread = threading.get_ident() not in thread_stacks
            if creating_new_thread:
                thread_stacks[threading.get_ident()] = list(dspy.settings.main_stack)

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
                    dspy.logger.error(
                        f"Error for example in dev set: \t\t {e}\n\twith inputs:\n\t\t{example.inputs()}\n\nStack trace:\n\t{traceback.format_exc()}"
                    )
                else:
                    dspy.logger.error(
                        f"Error for example in dev set: \t\t {e}. Set `provide_traceback=True` to see the stack trace."
                    )

                return example_idx, example, {}, 0.0
            finally:
                if creating_new_thread:
                    del thread_stacks[threading.get_ident()]

        devset = list(enumerate(devset))
        tqdm.tqdm._instances.clear()

        preds = {label: [] for label in self.labels}

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

        mcc, cm = self.get_matthews_corrcoef(preds, devset, return_cm=True)

        dspy.logger.info(f"MCC: {mcc:.6f}")

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
    def __init__(
            self,
            labels,
            teacher_settings={},
            max_bootstrapped_demos=4,
            max_labeled_demos=16,
            max_rounds=1,
            num_candidate_programs=16,
            num_threads=6,
            max_errors=10,
            stop_at_score=None,
            metric_threshold=None,
            use_class_weight=True,
            output_field="response",
    ):
        self.labels = labels
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
                labels=self.labels,
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
