import asyncio
import pytest
from unittest.mock import Mock, patch
import dspy
from dspy.primitives.program import Module


class SimpleAsyncModule(Module):
    async def aforward(self, x):
        await asyncio.sleep(0.1)  # Simulate async work
        return x * 2

    def forward(self, x):
        return x * 2

class SimpleSyncModule(Module):
    def forward(self, x):
        return x * 2

class MixedModule(Module):
    async def aforward(self, x, y=None):
        if y is not None:
            await asyncio.sleep(0.1)
            return x + y
        return x * 2
    
    def forward(self, x, y=None):
        if y is not None:
            return x + y
        return x * 2

@pytest.mark.asyncio
async def test_module_async_call():
    """Test that async arguments trigger aforward"""
    module = SimpleAsyncModule()
    async def async_input():
        await asyncio.sleep(0.1)
        return 5
    
    result = await module(async_input())
    assert result == 10

@pytest.mark.asyncio
async def test_module_sync_call():
    """Test that sync arguments use forward"""
    module = SimpleSyncModule()
    result = module(5)
    assert result == 10

@pytest.mark.asyncio
async def test_mixed_module_async():
    """Test mixed module with async arguments"""
    module = MixedModule()
    async def async_value():
        await asyncio.sleep(0.1)
        return 7
    result = await module(5, y=async_value())
    assert result == 12

@pytest.mark.asyncio
async def test_mixed_args():
    """Test handling of mixed sync/async arguments"""
    module = SimpleAsyncModule()
    async def async_value():
        await asyncio.sleep(0.1)
        return 5
    
    # Mix of sync and async args
    result = await module(async_value())
    assert result == 10

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in async context"""
    class ErrorModule(Module):
        async def aforward(self, x):
            raise ValueError("Test error")
        
        def forward(self, x):
            return x * 2

    module = ErrorModule()
    with pytest.raises(ValueError, match="Test error"):
        await module(5)

@pytest.mark.asyncio
async def test_callback_handling():
    """Test that callbacks work in async context"""
    class TestCallback(dspy.utils.callback.BaseCallback):
        def __init__(self):
            self.mock = Mock()
        
        def on_module_start(self, call_id, instance, inputs):
            self.mock.on_module_start(call_id, instance, inputs)
        
        def on_module_end(self, call_id, outputs, exception):
            self.mock.on_module_end(call_id, outputs, exception)

    callback = TestCallback()
    module = SimpleAsyncModule()
    module.callbacks.append(callback)
    
    result = await module(5)
    assert result == 10
    assert callback.mock.on_module_start.called
    assert callback.mock.on_module_end.called

def test_not_implemented():
    """Test NotImplementedError is raised when neither forward nor aforward is implemented"""
    module = Module()
    with pytest.raises(NotImplementedError):
        module(5) 

@pytest.mark.asyncio
async def test_async_type_detection():
    """Test detection of different async types"""
    module = SimpleAsyncModule()
    
    # Test coroutine
    async def coro():
        return 5
    result = await module(coro())
    assert result == 10
    
    # Test future
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future.set_result(5)
    result = await module(future)
    assert result == 10
    
    # Test custom awaitable
    class CustomAwaitable:
        def __await__(self):
            async def inner():
                return 5
            return inner().__await__()
    result = await module(CustomAwaitable())
    assert result == 10

def test_aforward_docstring():
    """Verify aforward has proper documentation"""
    doc = Module.aforward.__doc__
    
    # Check docstring exists and has key elements
    assert doc is not None, "aforward should have a docstring"
    
    # Check key implementation guidance
    assert "async def" in doc, "Should mention async def usage"
    assert "await" in doc, "Should mention await usage"
    assert "Example:" in doc, "Should include example code"
    
    # Check warning about blocking operations
    assert "blocking" in doc.lower(), "Should warn about blocking operations"
    
    # Check it has proper sections
    assert "Args:" in doc, "Should document arguments"
    assert "Returns:" in doc, "Should document return value"
    assert "Raises:" in doc, "Should document exceptions"

@pytest.mark.asyncio
async def test_edge_case_args():
    """Test edge cases in argument detection and handling"""
    class EdgeModule(Module):
        async def aforward(self, x, y=None):
            if x is None and y is None:
                return 0
            if x is None:
                return y
            if y is None:
                return x
            return x + y
        
        def forward(self, x, y=None):
            if x is None and y is None:
                return 0
            if x is None:
                return y
            if y is None:
                return x
            return x + y
    
    module = EdgeModule()
    
    # Test None arguments
    result = await module(None)
    assert result == 0
    
    # Test empty async value
    async def empty_async():
        return None
    result = await module(empty_async())
    assert result == 0
    
    # Test mixed None and async
    async def value_async():
        return 5
    result = await module(None, y=value_async())
    assert result == 5
    
    # Test multiple async args with None
    result = await module(
        empty_async(),
        y=value_async()
    )
    assert result == 5

@pytest.mark.asyncio
async def test_async_kwargs_only():
    """Test async detection in kwargs-only calls"""
    module = MixedModule()
    
    async def async_value():
        await asyncio.sleep(0.01)
        return 7
    
    # Call with only kwargs, no positional args
    result = await module(x=5, y=async_value())
    assert result == 12

    # Call with async value in first kwarg
    result = await module(x=async_value(), y=5)
    assert result == 12 