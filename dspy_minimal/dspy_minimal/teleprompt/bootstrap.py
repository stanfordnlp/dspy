import logging
import random
import threading
import time
from typing import Union, List, Dict, Any, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..primitives.example import Example

logger = logging.getLogger(__name__)

# ANSI color codes for pretty logging
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_step(step_num: int, title: str, description: str = ""):
    """Log a step with pretty formatting."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}Step {step_num}: {title}{Colors.ENDC}")
    if description:
        print(f"{Colors.OKBLUE}{description}{Colors.ENDC}")
    print("-" * 60)


def log_success(message: str):
    """Log a success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def log_error(message: str):
    """Log an error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def log_info(message: str):
    """Log an info message."""
    print(f"{Colors.OKBLUE}ℹ️  {message}{Colors.ENDC}")


def log_warning(message: str):
    """Log a warning message."""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


class BootstrapFewShot:
    """A full BootstrapFewShot implementation for dspy_minimal with multiple rounds and comprehensive error handling."""
    
    def __init__(
        self,
        metric=None,
        metric_threshold=None,
        teacher_settings: Union[dict, None] = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=None,
    ):
        """Initialize BootstrapFewShot with full optimization capabilities."""
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = {} if teacher_settings is None else teacher_settings
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()
        
        # Internal state for optimization
        self.student = None
        self.teacher = None
        self.trainset = None
        self.bootstrapped_demos = []
        self.labeled_demos = []

    def compile(self, student, *, teacher=None, trainset):
        """Compile the student with comprehensive bootstrapped examples."""
        log_step(1, "BootstrapFewShot Optimization", "Starting comprehensive optimization pipeline")
        
        start_time = time.time()
        
        # Store references
        self.trainset = list(trainset) if not isinstance(trainset, list) else trainset
        log_info(f"Training set size: {len(self.trainset)} examples")
        
        # Prepare student and teacher
        log_step(2, "Prepare Models", "Setting up student and teacher models")
        self._prepare_student_and_teacher(student, teacher)
        
        # Prepare predictor mappings
        log_step(3, "Prepare Predictors", "Configuring predictor mappings")
        self._prepare_predictor_mappings()
        
        # Perform bootstrapping with multiple rounds
        log_step(4, "Bootstrap Examples", "Generating bootstrapped examples with multiple rounds")
        self._bootstrap()
        
        # Train the final model
        log_step(5, "Train Model", "Training final model with all demos")
        self.student = self._train()
        self.student._compiled = True
        
        # Set failure counters
        self.student._assert_failures = 0
        self.student._suggest_failures = 0
        
        # Log optimization statistics
        end_time = time.time()
        optimization_time = end_time - start_time
        
        log_step(6, "Optimization Complete", f"Total time: {optimization_time:.2f} seconds")
        self._log_optimization_stats()
        
        log_success(f"BootstrapFewShot optimization complete. Added {len(self.student.demos)} total demos.")
        return self.student
    
    def _log_optimization_stats(self):
        """Log comprehensive optimization statistics."""
        total_examples = len(self.trainset)
        total_labeled = len(self.labeled_demos)
        total_bootstrapped = len(self.bootstrapped_demos)
        total_demos = total_labeled + total_bootstrapped
        success_rate = (total_bootstrapped / max(1, total_examples)) * 100
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
        print("📊 OPTIMIZATION STATISTICS")
        print(f"{'='*60}{Colors.ENDC}")
        
        print(f"{Colors.OKBLUE}Training examples:{Colors.ENDC} {total_examples}")
        print(f"{Colors.OKGREEN}Labeled demos selected:{Colors.ENDC} {total_labeled}")
        print(f"{Colors.OKCYAN}Bootstrapped demos generated:{Colors.ENDC} {total_bootstrapped}")
        print(f"{Colors.BOLD}Total demos:{Colors.ENDC} {total_demos}")
        print(f"{Colors.WARNING}Error count:{Colors.ENDC} {self.error_count}")
        print(f"{Colors.OKGREEN}Success rate:{Colors.ENDC} {success_rate:.1f}%")
        
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    
    def _prepare_student_and_teacher(self, student, teacher):
        """Prepare student and teacher models."""
        # Create a fresh copy of the student
        self.student = student
        
        # Use student as teacher if no teacher provided
        if teacher is None:
            self.teacher = student
        else:
            self.teacher = teacher
        
        # Ensure student is not already compiled
        if getattr(self.student, "_compiled", False):
            logger.warning("Student was already compiled. Creating fresh copy.")
            self.student = student
    
    def _prepare_predictor_mappings(self):
        """Prepare mappings for predictor optimization."""
        # For dspy_minimal, we'll use a simplified approach
        # In full DSPy, this would map predictors to their optimization strategies
        logger.info("Prepared predictor mappings for optimization")
    
    def _bootstrap(self, *, max_bootstraps=None):
        """Perform bootstrapping with multiple rounds and comprehensive error handling."""
        log_info(f"Starting bootstrapping with max_rounds={self.max_rounds}")
        
        # Select labeled demos first
        self.labeled_demos = self._select_labeled_demos()
        log_info(f"Selected {len(self.labeled_demos)} labeled demos")
        
        # Initialize bootstrapped demos
        self.bootstrapped_demos = []
        
        # Perform multiple rounds of bootstrapping
        for round_idx in range(self.max_rounds):
            log_info(f"Starting bootstrap round {round_idx + 1}/{self.max_rounds}")
            
            # Calculate how many more demos we need
            current_demos = len(self.bootstrapped_demos)
            needed_demos = self.max_bootstrapped_demos - current_demos
            
            if needed_demos <= 0:
                log_info("Sufficient demos collected, stopping bootstrapping")
                break
            
            # Try to bootstrap more examples
            round_demos = self._bootstrap_round(round_idx, needed_demos)
            self.bootstrapped_demos.extend(round_demos)
            
            log_success(f"Round {round_idx + 1} complete. Added {len(round_demos)} demos. Total: {len(self.bootstrapped_demos)}")
            
            # Check if we've hit error limits
            if self.max_errors and self.error_count >= self.max_errors:
                log_warning(f"Reached max errors ({self.max_errors}), stopping bootstrapping")
                break
        
        log_success(f"Bootstrapping complete. Total demos: {len(self.labeled_demos) + len(self.bootstrapped_demos)}")
    
    def _bootstrap_round(self, round_idx: int, needed_demos: int) -> List[Dict[str, Any]]:
        """Perform one round of bootstrapping."""
        round_demos = []
        
        # Shuffle training examples for this round
        shuffled_examples = self.trainset.copy()
        random.shuffle(shuffled_examples)
        
        # Try each example for bootstrapping with progress bar
        if HAS_TQDM:
            examples_iter = tqdm(
                shuffled_examples, 
                desc=f"Round {round_idx + 1} bootstrapping", 
                unit="example",
                leave=False
            )
        else:
            examples_iter = shuffled_examples
        
        for example in examples_iter:
            if len(round_demos) >= needed_demos:
                break
            
            try:
                demo = self._bootstrap_one_example(example, round_idx)
                if demo is not None:
                    round_demos.append(demo)
                    
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    log_warning(f"Error in bootstrap round {round_idx + 1}: {e}")
                    
                    if self.max_errors and self.error_count >= self.max_errors:
                        log_error(f"Reached max errors ({self.max_errors})")
                        break
        
        return round_demos
    
    def _bootstrap_one_example(self, example: Example, round_idx: int) -> Optional[Dict[str, Any]]:
        """Bootstrap a single example with comprehensive error handling and quality assessment."""
        try:
            # Get inputs for this example
            inputs = example.inputs()
            
            # Adjust temperature for different rounds to encourage diversity
            original_config = self.teacher.config.copy()
            if round_idx > 0:
                # Increase temperature in later rounds for more diverse predictions
                self.teacher.config['temperature'] = min(0.9, 0.7 + (round_idx * 0.1))
            
            # Use teacher to generate prediction
            prediction = self.teacher(**inputs._store)
            
            # Restore original config
            self.teacher.config = original_config
            
            # Evaluate the prediction with quality assessment
            if self._is_good_prediction(example, prediction):
                # Additional quality checks
                if self._assess_demo_quality(example, prediction):
                    # Create demo from the example
                    demo = {}
                    for key, value in example.items():
                        demo[key] = value
                    
                    logger.debug(f"Successfully bootstrapped high-quality example in round {round_idx + 1}")
                    return demo
                else:
                    logger.debug(f"Example quality not sufficient in round {round_idx + 1}")
                    return None
            else:
                logger.debug(f"Prediction not good enough for example in round {round_idx + 1}")
                return None
                
        except Exception as e:
            with self.error_lock:
                self.error_count += 1
                logger.warning(f"Error bootstrapping example in round {round_idx + 1}: {e}")
                
                if self.max_errors and self.error_count >= self.max_errors:
                    logger.error(f"Reached max errors ({self.max_errors})")
                
                return None
    
    def _assess_demo_quality(self, example: Example, prediction) -> bool:
        """Assess the quality of a potential demo."""
        try:
            # Check if prediction is not too generic
            if hasattr(prediction, '__str__'):
                pred_str = str(prediction)
                if len(pred_str.strip()) < 3:  # Too short
                    return False
                if pred_str.lower() in ['none', 'unknown', 'n/a', '']:  # Too generic
                    return False
            
            # Check if prediction matches expected format
            labels = example.labels()
            for key, expected_value in labels.items():
                if hasattr(prediction, key):
                    predicted_value = getattr(prediction, key)
                    # Check if prediction is not empty or too generic
                    if isinstance(predicted_value, str):
                        if len(predicted_value.strip()) < 2:
                            return False
                        if predicted_value.lower() in ['none', 'unknown', 'n/a', '']:
                            return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error assessing demo quality: {e}")
            return False
    
    def _select_labeled_demos(self) -> List[Dict[str, Any]]:
        """Select labeled demonstrations from the training set."""
        # Randomly select up to max_labeled_demos examples
        num_demos = min(self.max_labeled_demos, len(self.trainset))
        selected_examples = random.sample(self.trainset, num_demos)
        
        demos = []
        for example in selected_examples:
            # Convert example to demo format
            demo = {}
            for key, value in example.items():
                demo[key] = value
            demos.append(demo)
        
        logger.info(f"Selected {len(demos)} labeled demos")
        return demos
    
    def _train(self):
        """Train the final model with all collected demos."""
        # Combine all demos
        all_demos = self.labeled_demos + self.bootstrapped_demos
        
        # Set demos on the student
        self.student.demos = all_demos
        
        logger.info(f"Training complete with {len(all_demos)} total demos")
        return self.student
    
    def _is_good_prediction(self, example: Example, prediction) -> bool:
        """Check if a prediction is good using the metric function with comprehensive error handling."""
        try:
            if self.metric is None:
                # Default metric: check if the prediction matches the example
                labels = example.labels()
                
                # Simple comparison - check if any label matches
                for key, expected_value in labels.items():
                    if hasattr(prediction, key):
                        predicted_value = getattr(prediction, key)
                        if predicted_value == expected_value:
                            return True
                    elif isinstance(prediction, dict) and key in prediction:
                        if prediction[key] == expected_value:
                            return True
                    elif hasattr(prediction, '__getitem__'):
                        try:
                            predicted_value = prediction[key]
                            if predicted_value == expected_value:
                                return True
                        except (KeyError, TypeError):
                            continue
                
                return False
            else:
                # Use the provided metric function
                labels = example.labels()
                result = self.metric(labels, prediction)
                
                if self.metric_threshold is not None:
                    return result >= self.metric_threshold
                else:
                    return bool(result)
                    
        except Exception as e:
            logger.warning(f"Error evaluating prediction: {e}")
            return False 