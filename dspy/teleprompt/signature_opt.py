from .copro_optimizer import COPRO

"""
===============================================================
DEPRECATED!!!
PLEASE USE COPRO INSTEAD.
===============================================================

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


class SignatureOptimizer(COPRO):
    def __init__(
        self,
        prompt_model=None,
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    ):
        print(
            "\u001b[31m[WARNING] SignatureOptimizer has been deprecated and replaced with COPRO.  SignatureOptimizer will be removed in a future release. \u001b[31m",
        )
        super().__init__(prompt_model, metric, breadth, depth, init_temperature, verbose, track_stats)

    def compile(self, student, *, devset, eval_kwargs):
        return super().compile(student, trainset=devset, eval_kwargs=eval_kwargs)
