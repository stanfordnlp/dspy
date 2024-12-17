import random
from typing import List
import pandas as pd
import warnings
from tqdm import tqdm
import dspy
from dspy.predict.knn import KNN
from dspy.teleprompt import BootstrapFewShot
from .teleprompt import Teleprompter
import types
from dspy.predict.parallel import Parallel


class MetaKNNFewShot(Teleprompter):
    def __init__(
        self, 
        k: int,
        trainset,
        n_programs: int = 5,
        vectorizer=None,
        metric=None,
        metric_threshold=None,
        max_labeled_demos=16,
        max_rounds=1,
        max_bootstrapped_demos=8,
    ):
        """
        A Teleprompter that uses meta-learning with KNN to select the best few-shot prompts.
        
        Args:
            k: Number of nearest neighbors to consider
            trainset: List of training examples
            n_programs: Number of different random programs to generate
            vectorizer: Optional dspy.Embedder for computing similarities
            metric: Evaluation metric for comparing predictions
            metric_threshold: Optional threshold for the metric
            max_labeled_demos: Maximum number of labeled demonstrations
            max_rounds: Maximum rounds for bootstrapping
            max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        """
        self.KNN = KNN(k, trainset, vectorizer=vectorizer)
        self.n_programs = n_programs
        self.metric = metric
        self.metric_threshold = metric_threshold
        
        # Store bootstrap args for creating multiple programs
        self.bootstrap_args = {
            "metric": metric,
            "metric_threshold": metric_threshold,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "max_rounds": max_rounds,
        }
        
        self.programs = []
        self.performance_cache = pd.DataFrame()

    def compile(self, student, *, teacher=None, trainset=None, valset=None):
        """Compile the student program using meta-KNN few-shot learning."""
        student_copy = student.reset_copy()
        
        # Generate multiple programs using bootstrap
        self._generate_programs(student, teacher, trainset)
        
        # Create performance cache for all training examples
        self._build_performance_cache(trainset)

        def forward_pass(_, **kwargs):
            # Find similar examples using KNN
            similar_examples = self.KNN(**kwargs)
            
            # Get the best program based on performance on similar examples
            best_program = self._select_best_program(similar_examples)
            
            # Use the best program to generate the answer
            return best_program(**kwargs)

        student_copy.forward = types.MethodType(forward_pass, student_copy)
        return student_copy

    def _generate_programs(self, student, teacher, trainset):
        """Generate multiple different bootstrapped programs."""
        rng = random.Random(0)  # Use seeded RNG for reproducibility
        
        for prog_idx in tqdm(range(self.n_programs), desc="Generating programs"):
            # Create a shuffled copy of the trainset
            shuffled_trainset = list(trainset)
            rng.shuffle(shuffled_trainset)
            
            # Create bootstrap with a different seed for each program
            bootstrap = BootstrapFewShot(
                **self.bootstrap_args,
                teacher_settings={"temperature": 0.7 + 0.1 * prog_idx}  # Add temperature variation
            )
            
            program = bootstrap.compile(
                student.reset_copy(),
                teacher=teacher,
                trainset=shuffled_trainset
            )
            self.programs.append(program)

    def _build_performance_cache(self, trainset):
        """Build a cache of program performances on training examples in parallel."""
        parallel = Parallel(num_threads=32, disable_progress_bar=False)
        
        # Create execution pairs for each (program, example) combination
        exec_pairs = []
        for example in trainset:
            for prog_idx, program in enumerate(self.programs):
                # Skip if example is used as demo in this program
                if any(id(example) == id(demo) for predictor in program.predictors() for demo in predictor.demos):
                    continue
                    
                # Create pair with program and example
                exec_pairs.append((program, example))
        
        # Run parallel processing
        results = parallel(exec_pairs)
        
        # Build performance data from results
        performance_data = []
        result_idx = 0
        
        for example in trainset:
            row = {'example_id': id(example)}
            
            for prog_idx, program in enumerate(self.programs):
                if any(id(example) == id(demo) for predictor in program.predictors() for demo in predictor.demos):
                    row[f'program_{prog_idx}'] = float('nan')
                else:
                    try:
                        prediction = results[result_idx]
                        score = self.metric(example, prediction) if self.metric else 1.0
                        row[f'program_{prog_idx}'] = score
                        result_idx += 1
                    except Exception:
                        row[f'program_{prog_idx}'] = 0.0
                    
            performance_data.append(row)
        
        self.performance_cache = pd.DataFrame(performance_data)

    def _select_best_program(self, similar_examples):
        """Select the best program based on performance on similar examples."""
        example_ids = [id(ex) for ex in similar_examples]
        relevant_scores = self.performance_cache[
            self.performance_cache.example_id.isin(example_ids)
        ]
        
        # Calculate average performance for each program, ignoring NaN values
        program_scores = relevant_scores.drop('example_id', axis=1).mean(skipna=True)
        
        # Suppress FutureWarning for this specific operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            program_scores = pd.to_numeric(program_scores.fillna(0.0), downcast=None)
        
        best_program_idx = int(program_scores.idxmax().split('_')[1])
        
        return self.programs[best_program_idx] 