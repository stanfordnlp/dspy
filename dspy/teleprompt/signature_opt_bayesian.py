from dspy.teleprompt.mipro_optimizer import MIPRO
import warnings

"""
===============================================================
DEPRECATED!!!
PLEASE USE MIPRO INSTEAD.
===============================================================

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

class BayesianSignatureOptimizer(MIPRO):
    def __init__(self, prompt_model=None, task_model=None, teacher_settings={}, n=10, metric=None, init_temperature=1.0, verbose=False, track_stats=True, view_data_batch_size=10):
        # warnings.warn(
        #         "`BayesianSignatureOptimizer` is deprecated and will be removed in a future version. "
        #         "Use `MIPRO` instead.", 
        #         DeprecationWarning
        #     )
        print(u"\u001b[31m[WARNING] BayesianSignatureOptimizer has been deprecated and replaced with MIPRO.  BayesianSignatureOptimizer will be removed in a future release. \u001b[31m")

        super().__init__(prompt_model, task_model, teacher_settings,n,metric,init_temperature,verbose,track_stats,view_data_batch_size)

    def compile(self, student, *, devset, max_bootstrapped_demos, max_labeled_demos, eval_kwargs, seed=42, view_data=True, view_examples=True, requires_permission_to_run=True, trials_num=None, optuna_trials_num=None):
        # Define ANSI escape codes for colors
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'  # Resets the color to default
        
        # Check if both trials_num and optuna_trials_num are None
        if trials_num is None and optuna_trials_num is None:
            raise ValueError(f"{YELLOW}{BOLD}You must specify the number of trials using the 'trials_num' parameter.{ENDC}")

        # Check if the deprecated parameter is used
        if optuna_trials_num is not None:
            # Issue a deprecation warning
            warnings.warn(
                "`optuna_trials_num` is deprecated and will be removed in a future version. "
                "Use `trials_num` instead.", 
                DeprecationWarning
            )
            # Use trials_num as a fallback if trials_num is not provided
            if trials_num is None:
                trials_num = optuna_trials_num
        return super().compile(student, trainset=devset, max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos, eval_kwargs=eval_kwargs, seed=seed, view_data=view_data, view_examples=view_examples, requires_permission_to_run=requires_permission_to_run, num_trials=trials_num)