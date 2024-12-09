import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class MaxScoreOptimizer:
    def __init__(self, num_demos=5, num_attempts=100):
        self.num_demos = num_demos
        self.num_attempts = num_attempts

    def try_prediction(self, original_forward, demos, args, kwargs):
        """Single prediction attempt with given demos"""
        try:
            result = original_forward(*args, **kwargs)
            if result.answer == kwargs.get('answer'):
                return True, result
            return False, result
        except Exception:
            return False, None

    def compile(self, program, trainset, evalset=None, **kwargs):
        """
        For each test example, try different random few-shot demos multiple times
        and consider it correct if any attempt succeeds.
        """
        program = program.deepcopy()
        original_forward = program.forward
        trainset = list(trainset)
        print(f"Using full training set of size: {len(trainset)}")

        # Capture optimizer attributes and methods in closure
        num_demos = self.num_demos
        num_attempts = self.num_attempts
        try_prediction = self.try_prediction  # Capture the method

        def new_forward(self, *args, **kwargs):
            best_result = None

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []

                for _ in range(num_attempts):
                    demos = random.sample(trainset, num_demos)
                    self.demos = demos

                    future = executor.submit(
                        try_prediction,  # Use captured method
                        original_forward,
                        demos,
                        args,
                        kwargs
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    success, result = future.result()
                    if success:
                        return result
                    if result is not None and best_result is None:
                        best_result = result

            return best_result if best_result is not None else original_forward(*args, **kwargs)

        program.forward = new_forward.__get__(program)
        return program


def max_score_optimizer_caller(default_program, trainset, devset, test_name, dataset_name, kwargs):
    optimizer = MaxScoreOptimizer(num_demos=5, num_attempts=100)

    compiled_program = optimizer.compile(
        default_program,
        trainset=trainset,
        evalset=devset
    )

    output = {
        "test_name": f"{dataset_name}_{test_name}_max_score",
        "additional_notes": "Tests best possible score with 5 random demos from full training set, 100 attempts per example",
        "fewshot_candidates_num": 5,
        "max_bootstrapped_demos": 100
    }

    return compiled_program, output