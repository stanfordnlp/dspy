import random
import hashlib
import json
from typing import Any, Dict, List, Optional, Callable
from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""

class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False, seed=None):
        """A common reduce_fn is dspy.majority."""
        # FIXED: Removed assertion that blocked deterministic ensembles
        # assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."
        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic
        self.seed = seed

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn
        deterministic = self.deterministic
        seed = self.seed  # FIXED: Added this line to define seed variable

        import dspy

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs
                self._deterministic = deterministic
                self._size = size
                self._seed = seed
                self._program_selection_cache = {}

            def _hash_example(self, *args, **kwargs) -> str:
                """Create a deterministic hash for the input example."""
                try:
                    # Convert args and kwargs to a serializable format
                    hashable_data = {
                        'args': [str(arg) for arg in args],
                        'kwargs': {k: str(v) for k, v in kwargs.items()}
                    }
                    
                    # Create deterministic string representation
                    data_str = json.dumps(hashable_data, sort_keys=True, ensure_ascii=True)
                    
                    # Generate hash
                    return hashlib.md5(data_str.encode('utf-8')).hexdigest()
                except Exception:
                    # Fallback: use string representation
                    fallback_str = f"{args}_{sorted(kwargs.items())}"
                    return hashlib.md5(fallback_str.encode('utf-8')).hexdigest()

            def _select_programs_deterministic(self, example_hash: str) -> List:
                """Select programs deterministically based on example hash."""
                if example_hash in self._program_selection_cache:
                    return self._program_selection_cache[example_hash]

                if not self._size or self._size >= len(self.programs):
                    selected = self.programs
                else:
                    # Use hash as seed for deterministic selection
                    hash_int = int(example_hash[:8], 16)  # Use first 8 hex chars
                    
                    # Combine with global seed if provided
                    if self._seed is not None:
                        hash_int = hash_int * self._seed
                    
                    local_random = random.Random(hash_int)
                    selected = local_random.sample(self.programs, self._size)

                # Cache the selection
                self._program_selection_cache[example_hash] = selected
                return selected

            def forward(self, *args, **kwargs):
                if self._deterministic:
                    # Deterministic mode: use example hashing
                    example_hash = self._hash_example(*args, **kwargs)
                    programs = self._select_programs_deterministic(example_hash)
                else:
                    # Original random sampling behavior
                    programs = random.sample(self.programs, size) if size else self.programs

                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)
                return outputs

        return EnsembledProgram()

# Enhanced version with additional features for better deterministic support
class EnhancedEnsemble(Ensemble):
    """Enhanced Ensemble with better deterministic support and additional features."""
    
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False,
                 seed=None, cache_selections=True, hash_algorithm='md5'):
        """
        Enhanced ensemble with better deterministic support.
        
        Args:
            reduce_fn: Function to reduce ensemble outputs (e.g., dspy.majority)
            size: Number of programs to sample for each prediction
            deterministic: Whether to use deterministic program selection
            seed: Global seed for deterministic behavior (optional)
            cache_selections: Whether to cache program selections for efficiency
            hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        """
        super().__init__(reduce_fn=reduce_fn, size=size, deterministic=deterministic, seed=seed)
        self.cache_selections = cache_selections
        self.hash_algorithm = hash_algorithm
        
        # Set global seed if provided
        if seed is not None and deterministic:
            random.seed(seed)

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn
        deterministic = self.deterministic
        seed = self.seed  # FIXED: Added this line to define seed variable
        cache_selections = self.cache_selections
        hash_algorithm = self.hash_algorithm

        import dspy

        class EnhancedEnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs
                self._deterministic = deterministic
                self._size = size
                self._seed = seed
                self._cache_selections = cache_selections
                self._hash_algorithm = hash_algorithm
                self._program_selection_cache = {} if cache_selections else None
                self._stats = {
                    'total_predictions': 0,
                    'cache_hits': 0,
                    'unique_selections': 0
                }

            def _hash_example(self, *args, **kwargs) -> str:
                """Create a deterministic hash for the input example."""
                try:
                    # More robust serialization
                    hashable_data = {}
                    
                    # Process args
                    if args:
                        hashable_data['args'] = []
                        for arg in args:
                            if hasattr(arg, '__dict__'):
                                hashable_data['args'].append(str(vars(arg)))
                            else:
                                hashable_data['args'].append(str(arg))
                    
                    # Process kwargs
                    if kwargs:
                        hashable_data['kwargs'] = {}
                        for k, v in kwargs.items():
                            if hasattr(v, '__dict__'):
                                hashable_data['kwargs'][k] = str(vars(v))
                            else:
                                hashable_data['kwargs'][k] = str(v)
                    
                    # Create deterministic string
                    data_str = json.dumps(hashable_data, sort_keys=True, ensure_ascii=True)
                    
                    # Generate hash using specified algorithm
                    if self._hash_algorithm == 'sha1':
                        return hashlib.sha1(data_str.encode('utf-8')).hexdigest()
                    elif self._hash_algorithm == 'sha256':
                        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
                    else:  # default to md5
                        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
                        
                except Exception as e:
                    # More robust fallback
                    fallback_str = f"{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
                    return hashlib.md5(fallback_str.encode('utf-8')).hexdigest()

            def _select_programs_deterministic(self, example_hash: str) -> List:
                """Select programs deterministically based on example hash."""
                # Check cache first
                if self._cache_selections and example_hash in self._program_selection_cache:
                    self._stats['cache_hits'] += 1
                    return self._program_selection_cache[example_hash]

                if not self._size or self._size >= len(self.programs):
                    selected = self.programs
                else:
                    # Use hash as seed for deterministic selection
                    hash_int = int(example_hash[:8], 16)  # Use first 8 hex chars
                    
                    # Combine with global seed if provided
                    if self._seed is not None:
                        hash_int = hash_int ^ self._seed
                    
                    local_random = random.Random(hash_int)
                    selected = local_random.sample(self.programs, self._size)

                # Cache the selection
                if self._cache_selections:
                    self._program_selection_cache[example_hash] = selected
                    self._stats['unique_selections'] = len(self._program_selection_cache)

                return selected

            def forward(self, *args, **kwargs):
                self._stats['total_predictions'] += 1
                
                if self._deterministic:
                    # Deterministic mode: use example hashing
                    example_hash = self._hash_example(*args, **kwargs)
                    programs = self._select_programs_deterministic(example_hash)
                else:
                    # Original random sampling behavior
                    programs = random.sample(self.programs, size) if size else self.programs

                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)
                return outputs

            def get_stats(self) -> Dict[str, Any]:
                """Get ensemble statistics."""
                stats = self._stats.copy()
                if self._cache_selections:
                    stats['cache_hit_rate'] = (
                        self._stats['cache_hits'] / max(self._stats['total_predictions'], 1)
                    )
                return stats

            def clear_cache(self):
                """Clear the program selection cache."""
                if self._cache_selections:
                    self._program_selection_cache.clear()
                    self._stats['cache_hits'] = 0
                    self._stats['unique_selections'] = 0

        return EnhancedEnsembledProgram()

# Utility function for your immediate use case
def create_deterministic_ensemble(programs: List, reduce_fn: Callable = None,
                                 size: Optional[int] = None, seed: int = 42):
    """
    Create a deterministic ensemble for immediate use.
    This is a convenience function that wraps the enhanced ensemble
    with sensible defaults for deterministic behavior.
    """
    ensemble = EnhancedEnsemble(
        reduce_fn=reduce_fn,
        size=size,
        deterministic=True,
        seed=seed,
        cache_selections=True,
        hash_algorithm='md5'
    )
    
    return ensemble.compile(programs)

# # Example usage for your specific case:
# def fix_your_ensemble_code():
#     """
#     Example of how to modify your existing code to use deterministic ensemble.
#     """
#     # Your original code with the fix:
#     ensemble_optimizer = EnhancedEnsemble(
#         reduce_fn=custom_majority_vote,  # Use your custom voting function
#         size=len(ensemble_programs),     # Number of models
#         deterministic=True,              # Now supported!
#         seed=42                          # For reproducible results
#     )
#     
#     # Compile the ensemble
#     ensemble_model = ensemble_optimizer.compile(ensemble_programs)
#     return ensemble_model
