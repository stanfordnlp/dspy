import magicattr
import inspect
import asyncio
from typing import Any, Union, Awaitable, TypeVar, Optional, List, Callable

from dspy.predict.parallel import Parallel
from dspy.primitives.module import BaseModule
from dspy.utils.callback import with_callbacks

T = TypeVar('T')


class ProgramMeta(type):
    pass


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False

    @with_callbacks
    def __call__(self, *args: Any, **kwargs: Any) -> Union[T, Awaitable[T]]:
        """Call the module with given arguments.
        
        Automatically determines whether to use sync or async execution based on arguments.
        If any argument is a coroutine, awaitable, or future, uses async execution.
        Also uses async execution if the module has a custom aforward implementation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Either the direct result (sync) or an awaitable of the result (async)
        """
        def is_async_arg(arg):
            return (inspect.iscoroutine(arg) or 
                   inspect.isawaitable(arg) or 
                   isinstance(arg, asyncio.Future))

        # Check if we should use async execution
        use_async = (
            # If any argument is async
            any(is_async_arg(arg) for arg in args) or
            any(is_async_arg(v) for v in kwargs.values()) or
            # Or if we have a custom aforward implementation
            (hasattr(self, 'aforward') and 
             self.aforward.__func__ is not Module.aforward)
        )

        if use_async:
            async def _async_call():
                # Process args concurrently
                args_coros = []
                args_indices = []
                resolved_args = list(args)  # Create modifiable copy
                
                # Identify async args and their positions
                for idx, arg in enumerate(resolved_args):
                    if is_async_arg(arg):
                        args_coros.append(arg)
                        args_indices.append(idx)
                
                # Resolve async args concurrently
                if args_coros:
                    args_results = await asyncio.gather(*args_coros)
                    for i, result in zip(args_indices, args_results):
                        resolved_args[i] = result  # Replace with resolved values
                
                # Process kwargs concurrently
                async_kwargs = {}
                kwarg_coros = []
                kwarg_keys = []
                
                # Separate async and sync kwargs
                for k, v in kwargs.items():
                    if is_async_arg(v):
                        kwarg_coros.append(v)
                        kwarg_keys.append(k)
                    else:
                        async_kwargs[k] = v
                
                # Resolve async kwargs concurrently
                if kwarg_coros:
                    kwarg_results = await asyncio.gather(*kwarg_coros)
                    async_kwargs.update(zip(kwarg_keys, kwarg_results))
                
                return await self.aforward(*resolved_args, **async_kwargs)
            return _async_call()
        
        # Use sync execution
        return self.forward(*args, **kwargs)

    async def aforward(self, *args: Any, **kwargs: Any) -> T:
        """Async version of forward.
        
        This method should be implemented by subclasses to provide async execution.
        By default, raises NotImplementedError to encourage proper async implementation.
        
        When implementing this method:
        1. Use 'async def' and 'await' for async operations
        2. Avoid blocking operations - they should be properly awaited
        3. Consider using asyncio.create_task for concurrent operations
        4. Be mindful of async context managers (use 'async with')
        
        Example:
            ```python
            class MyAsyncModule(Module):
                async def aforward(self, x):
                    # Good: proper async operation
                    result = await async_operation(x)
                    return result
                    
                    # Bad: blocking operation
                    # time.sleep(1)  # Don't do this!
                    
                    # Bad: sync operation without proper async
                    # return self.forward(x)  # Don't do this!
            ```
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The result of the async computation
            
        Raises:
            NotImplementedError: Subclasses must implement this method for async operations
        """
        raise NotImplementedError(
            "Subclasses must implement aforward for async operations. "
            "Do not use sync operations or blocking calls in this method."
        )

    def forward(self, *args: Any, **kwargs: Any) -> T:
        """Synchronous forward pass.
        
        Must be implemented by subclasses to define the module's computation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The result of the computation
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    # def activate_assertions(self, handler=backtrack_handler, **handler_args):
    #     """
    #     Activates assertions for the module.
    #     The default handler is the backtrack_handler.
    #     """
    #     assert_transform_module(self, handler, **handler_args)
    #     return self

    # def __deepcopy__(self, memo):
    #     # memo is a dict of id's to copies already made during the current call
    #     # Check if the object is already copied
    #     if id(self) in memo:
    #         return memo[id(self)]

    #     print(f"Deep copying {self.__class__.__name__}...")

    #     new_copy = copy.copy(self)
    #     memo[id(self)] = new_copy

    #     for k, v in self.__dict__.items():
    #         print(f"Copying attribute {k} of type {type(v)}...")
    #         setattr(new_copy, k, copy.deepcopy(v, memo))
    #         print("Done")

    #     return new_copy

    def batch(
        self,
        examples,
        num_threads: int = 32,
        max_errors: int = 10,
        return_failed_examples: bool = False,
        provide_traceback: bool = False,
        disable_progress_bar: bool = False,
    ):
        """
        Processes a list of dspy.Example instances in parallel using the Parallel module.

        :param examples: List of dspy.Example instances to process.
        :param batch_size: Number of threads to use for parallel processing.
        :param max_errors: Maximum number of errors allowed before stopping execution.
        :param return_failed_examples: Whether to return failed examples and exceptions.
        :param provide_traceback: Whether to include traceback information in error logs.
        :return: List of results, and optionally failed examples and exceptions.
        """
        # Create a list of execution pairs (self, example)
        exec_pairs = [(self, example.inputs()) for example in examples]

        # Create an instance of Parallel
        parallel_executor = Parallel(
            num_threads=num_threads,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        # Execute the forward method of Parallel
        if return_failed_examples:
            results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
            return results, failed_examples, exceptions
        else:
            results = parallel_executor.forward(exec_pairs)
            return results


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)


Program = Module
