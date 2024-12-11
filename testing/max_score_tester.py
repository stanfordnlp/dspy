import random
from typing import List, Dict, Optional, Any
import pandas as pd
from tqdm import tqdm
import os
import openai
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from abc import ABC, abstractmethod

class BaseMaxScoreTester(ABC):
    def __init__(
        self,
        num_threads: int = 32,
        early_stopping_threshold: float = 0.95,
        max_errors: int = 10,
        prompt_model_name: str = "gpt-3.5-turbo-1106",
        task_model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        prompt_model=None,
        task_model=None,
        colbert_v2_endpoint: str = "http://20.102.90.50:2017/wiki17_abstracts",
        dataset_name: str = "default",
    ):
        """
        Base class for testing maximum achievable scores on datasets.
        
        Args:
            num_threads: Number of threads for parallel evaluation
            early_stopping_threshold: Stop if we achieve this score
            max_errors: Maximum number of errors before stopping evaluation
            prompt_model_name: Name of the model to use for prompting
            task_model_name: Name of the model to use for task execution
            prompt_model: Optional custom prompt model
            task_model: Optional custom task model
            colbert_v2_endpoint: Endpoint for ColBERTv2 service
            dataset_name: Name of the dataset being tested
        """
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.early_stopping_threshold = early_stopping_threshold
        self.PROMPT_MODEL_NAME = prompt_model_name
        self.TASK_MODEL_NAME = task_model_name
        self.COLBERT_V2_ENDPOINT = colbert_v2_endpoint
        self.dataset_name = dataset_name
        
        # Initialize models
        self._setup_models(prompt_model, task_model)
        
        self.programs = []
        self.performance_cache = pd.DataFrame()

    def _setup_models(self, prompt_model, task_model):
        """Setup the language models and retrieval model."""
        load_dotenv()  # Load environment variables

        # Setup OpenAI
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("OPENAI_API_BASE"):
            openai.api_base = os.environ.get("OPENAI_API_BASE")

        # Prompt generation model
        if not prompt_model:
            self.prompt_model = dspy.OpenAI(model=self.PROMPT_MODEL_NAME, max_tokens=700)
        else:
            self.prompt_model = prompt_model

        # Task execution model
        if not task_model:
            self.task_model = dspy.HFClientTGI(
                model=self.TASK_MODEL_NAME,
                port=[7140, 7141, 7142, 7143],
                max_tokens=150,
            )
        else:
            self.task_model = task_model

        # Setup retrieval model
        self.colbertv2 = dspy.ColBERTv2(url=self.COLBERT_V2_ENDPOINT)
        
        # Configure DSPy settings
        dspy.settings.configure(rm=self.colbertv2, lm=self.task_model)

    def test_dataset(self, task: Any, save_excel: bool = True, **kwargs) -> Dict:
        """
        Test a dataset to find maximum achievable score.
        
        Args:
            task: Task object containing the dataset and metric
            **kwargs: Additional arguments passed to program generation
        
        Returns:
            Dict containing results and statistics
        """
        # Generate multiple programs
        self._generate_programs(task, **kwargs)
        
        # Evaluate programs on different splits
        train_results = self._evaluate_split(task.get_trainset(), task.get_metric(), "train")
        dev_results = self._evaluate_split(task.get_devset(), task.get_metric(), "dev")
        test_results = self._evaluate_split(task.get_testset(), task.get_metric(), "test")
        
        if save_excel:
            # Create dictionary of dataframes
            dfs = {
            'train': train_results['performance_df'],
            'dev': dev_results['performance_df'],
            'test': test_results['performance_df']
            }
        
            # Save to Excel
            pd.concat(dfs.values(), keys=dfs.keys()).to_excel(f'performance_{self.dataset_name}.xlsx')
        
        return {
            "train_results": train_results,
            "dev_results": dev_results,
            "test_results": test_results,
        }

    @abstractmethod
    def _generate_programs(self, task: Any, **kwargs) -> None:
        """Generate multiple different programs. Must be implemented by subclasses."""
        pass

    def _evaluate_split(self, dataset, metric, split_name):
        """
        Evaluate programs on dataset items in parallel until one succeeds.
        
        For each dataset item:
        - Try each program until one passes the metric
        - Stop early for that item once a program succeeds
        - Record results in a performance matrix (items × programs)
        """
        # Create parallel executor
        parallel = dspy.Parallel(
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            disable_progress_bar=False,
            provide_traceback=True,
        )
        
        class ItemProcessor:
            def __init__(self, programs, metric, dataset_name):
                self.programs = programs
                self.metric = metric
                self.dataset_name = dataset_name
            
            def discrete_retrieval_eval(self, example, pred, trace=None):
                gold_titles = set(
                    map(
                        dspy.evaluate.normalize_text,
                        [doc["key"] for doc in example["supporting_facts"]],
                    )
                )
                found_titles = set(
                    map(
                        dspy.evaluate.normalize_text,
                        [c.split(" | ")[0] for c in pred.retrieved_docs],
                    )
                )
                return gold_titles, found_titles
            
            def __call__(self, item_idx, item):
                """Process a single dataset item with all programs until success."""
                item_results = {
                    'item_idx': item_idx,
                    'split': split_name,
                    'input': str(item.inputs()),
                    'expected': str(item),
                    'successful_prediction': None,  # Will store the first successful prediction
                    'successful_program_idx': -1    # Will store index of first successful program
                }
                
                # Try each program until success
                for prog_idx, program in enumerate(self.programs):
                    try:
                        # Run the program on the item
                        prediction = program(**item.inputs())
                        
                        # Calculate score using metric
                        raw_score = self.metric(item, prediction)
                        score = raw_score[0] if isinstance(raw_score, tuple) else raw_score
                        score = float(score)
                        
                        # Record result with prediction
                        item_results[f'program_{prog_idx}'] = score
                        item_results[f'prediction_{prog_idx}'] = str(prediction)
                        
                        if self.dataset_name == "hover_retrieve_discrete":
                            gold_titles, found_titles = self.discrete_retrieval_eval(item, prediction)
                            item_results[f'gold_titles_{prog_idx}'] = gold_titles
                            item_results[f'gold_titles_len_{prog_idx}'] = len(gold_titles)
                            item_results[f'found_titles_{prog_idx}'] = found_titles
                            item_results[f'found_titles_len_{prog_idx}'] = len(found_titles)
                            item_results[f'gold_titles_issubset_found_titles_{prog_idx}'] = gold_titles.issubset(found_titles)
                            item_results[f'gold_titles_intersection_found_titles_{prog_idx}'] = gold_titles.intersection(found_titles)
                            item_results[f'gold_titles_intersection_found_titles_len_{prog_idx}'] = len(gold_titles.intersection(found_titles))
                            item_results[f'gold_titles_diff_found_titles_{prog_idx}'] = gold_titles.difference(found_titles)
                            item_results[f'gold_titles_diff_found_titles_len_{prog_idx}'] = len(gold_titles.difference(found_titles))
                            item_results[f'found_titles_diff_gold_titles_{prog_idx}'] = found_titles.difference(gold_titles)
                            item_results[f'found_titles_diff_gold_titles_len_{prog_idx}'] = len(found_titles.difference(gold_titles))
                        
                        # Check if this program passed and it's our first success
                        if score > 0 and item_results['successful_prediction'] is None:
                            item_results['successful_prediction'] = str(prediction)
                            item_results['successful_program_idx'] = prog_idx
                            break
                            
                    except Exception as e:
                        item_results[f'program_{prog_idx}'] = 0.0
                        item_results[f'prediction_{prog_idx}'] = str(e)
                
                return item_results
        
        # Create processor instance
        processor = ItemProcessor(self.programs, metric, dataset_name=self.dataset_name)
        
        # Create list of (idx, item) pairs
        items_with_idx = list(enumerate(dataset))
        
        # Process items in parallel
        exec_pairs = [(processor, (idx, item)) for idx, item in items_with_idx]
        results = parallel(exec_pairs)
        
        # Convert results to DataFrame
        performance_df = pd.DataFrame(results)
        
        # Add intersection of gold_titles_diff_found_titles if they exist
        if 'gold_titles_diff_found_titles_1' in performance_df.columns:
            # Get all columns containing gold_titles_diff_found_titles
            diff_columns = [col for col in performance_df.columns if 'gold_titles_diff_found_titles' in col]
            
            # Calculate intersection of differences across all programs for each row
            performance_df['common_missed_titles'] = performance_df[diff_columns].apply(
                lambda row: set.intersection(*[s for s in row if isinstance(s, set)]) if any(isinstance(s, set) for s in row) else set(),
                axis=1
            )
        
        # Add to performance cache
        self.performance_cache = pd.concat([
            self.performance_cache,
            performance_df
        ], ignore_index=True)
        
        # Calculate and display statistics
        program_columns = [col for col in performance_df.columns if col.startswith('program_')]

        # Calculate overall statistics
        solved_items = performance_df[program_columns].apply(
            lambda x: (x > 0).any(), axis=1
        ).mean()
        
        print(f"\n{split_name} Items Solved: {solved_items:.2%}")
        
        return {
            "solved_items": solved_items,
            "performance_df": performance_df
        }


class BootstrapMaxScoreTester(BaseMaxScoreTester):
    def __init__(
        self,
        n_programs: int = 10,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        num_threads: int = 32,
        early_stopping_threshold: float = 0.95,
        max_errors: int = 100,
        prompt_model_name: str = "gpt-3.5-turbo-1106",
        task_model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        prompt_model=None,
        task_model=None,
        colbert_v2_endpoint: str = "http://20.102.90.50:2017/wiki17_abstracts",
        dataset_name: str = "default",
    ):
        super().__init__(
            num_threads=num_threads,
            early_stopping_threshold=early_stopping_threshold,
            max_errors=max_errors,
            prompt_model_name=prompt_model_name,
            task_model_name=task_model_name,
            prompt_model=prompt_model,
            task_model=task_model,
            colbert_v2_endpoint=colbert_v2_endpoint,
            dataset_name=dataset_name,
        )
        
        self.n_programs = n_programs
        # Store bootstrap args
        self.bootstrap_args = {
            "max_bootstrapped_demos": 1,
            "max_labeled_demos": max_labeled_demos,
            "max_rounds": max_rounds,
        }

    def _generate_programs(self, task, teacher_model=None, **kwargs):
        """Generate multiple different bootstrapped programs."""
        rng = random.Random(0)  # Use seeded RNG for reproducibility
        student = task.get_program()
        
        for prog_idx in tqdm(range(self.n_programs), desc="Generating programs"):
            # Create a shuffled copy of the trainset
            shuffled_trainset = list(task.get_trainset())
            rng.shuffle(shuffled_trainset)
            
            # Create bootstrap with a different seed for each program
            bootstrap = BootstrapFewShot(
                **self.bootstrap_args,
                metric=task.get_metric(),
                teacher_settings={"temperature": 0.7 + 0.1 * prog_idx}
            )
            
            program = bootstrap.compile(
                student.reset_copy(),
                teacher=teacher_model,
                trainset=shuffled_trainset
            )
            self.programs.append(program)


class SimpleMaxScoreTester(BaseMaxScoreTester):
    """A simpler tester that just varies temperature and few-shot examples."""
    
    def __init__(
        self,
        n_programs: int = 10,
        num_threads: int = 32,
        early_stopping_threshold: float = 0.95,
        max_errors: int = 100,
    ):
        super().__init__(
            num_threads=num_threads,
            early_stopping_threshold=early_stopping_threshold,
            max_errors=max_errors,
        )
        self.n_programs = n_programs
    
    def _generate_programs(self, task, teacher_model=None, **kwargs):
        """Generate programs by varying temperature and few-shot examples."""
        rng = random.Random(0)
        student = task.get_program()
        trainset = task.get_trainset()
        
        # Generate programs with different temperatures and example combinations
        for prog_idx in tqdm(range(self.n_programs), desc="Generating programs"):
            temperature = 0.7 + 0.1 * (prog_idx % 5)  # Vary temperature between 0.7 and 1.1
            
            # Randomly select few-shot examples
            n_examples = min(rng.randint(2, 8), len(trainset))
            examples = rng.sample(trainset, n_examples)
            
            # Create a new program with these settings
            program = student.reset_copy()
            for predictor in program.predictors():
                predictor.demos = examples
                predictor.temperature = temperature
            
            self.programs.append(program) 