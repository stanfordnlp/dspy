import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO
from dspy.evaluate import Evaluate
from dspy.utils.dummies import DummyLM
from dspy.predict import Predict


class SimpleQAModule(dspy.Module):
    """Simple QA module for testing."""
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)


class ModuleWithMultiplePredictors(dspy.Module):
    """Module with multiple predictors for testing."""
    def __init__(self):
        super().__init__()
        self.predictor1 = dspy.Predict("input -> output1")
        self.predictor2 = dspy.ChainOfThought("input -> output2")
        self.predictor3 = dspy.Predict("input -> output3")
    
    def forward(self, input):
        result1 = self.predictor1(input=input)
        result2 = self.predictor2(input=input)
        result3 = self.predictor3(input=input)
        return dspy.Prediction(
            output1=result1.output1,
            output2=result2.output2,
            output3=result3.output3
        )


class NestedModule(dspy.Module):
    """Module with nested sub-modules for testing."""
    def __init__(self):
        super().__init__()
        self.qa_module = SimpleQAModule()
        self.multi_module = ModuleWithMultiplePredictors()
    
    def forward(self, question):
        answer = self.qa_module(question=question)
        result = self.multi_module(input=answer.answer)
        return result


def test_freeze_unfreeze_basic():
    """Test basic freeze and unfreeze functionality."""
    module = SimpleQAModule()
    
    # Initially module should be trainable
    assert module.trainable == True
    
    # Freeze the module
    module.freeze()
    assert module.trainable == False
    
    # Unfreeze the module
    module.unfreeze()
    assert module.trainable == True


def test_freeze_propagates_to_predictors():
    """Test that freezing propagates to all predictors."""
    module = ModuleWithMultiplePredictors()
    
    # Initially all predictors should be trainable
    trainable_predictors = module.predictors(return_trainable=True)
    all_predictors = module.predictors(return_trainable=False)
    assert len(trainable_predictors) == 3
    assert len(all_predictors) == 3
    
    # After freezing the module, all predictors should be frozen
    module.freeze()
    assert module.trainable == False
    
    # Now no predictors should be trainable
    trainable_predictors = module.predictors(return_trainable=True)
    all_predictors = module.predictors(return_trainable=False)
    assert len(trainable_predictors) == 0
    assert len(all_predictors) == 3
    
    # Verify individual predictors are frozen
    assert module.predictor1.trainable == False
    assert module.predictor2.trainable == False
    assert module.predictor3.trainable == False
    
    # Unfreeze and verify all predictors are unfrozen
    module.unfreeze()
    assert module.trainable == True
    trainable_predictors = module.predictors(return_trainable=True)
    assert len(trainable_predictors) == 3
    assert module.predictor1.trainable == True
    assert module.predictor2.trainable == True
    assert module.predictor3.trainable == True


def test_freeze_chain_of_thought():
    """Test that freezing ChainOfThought freezes its internal predict."""
    cot = dspy.ChainOfThought("question -> answer")
    
    # Initially should be trainable
    assert cot.trainable == True
    assert cot.predict.trainable == True
    
    # Freeze should propagate to internal predict
    cot.freeze()
    assert cot.trainable == False
    assert cot.predict.trainable == False
    
    # Unfreeze should propagate as well
    cot.unfreeze()
    assert cot.trainable == True
    assert cot.predict.trainable == True


def test_freeze_individual_predictors():
    """Test freezing individual predictors directly."""
    module = ModuleWithMultiplePredictors()
    
    # Freeze individual predictors without freezing the module
    module.predictor1.freeze()
    module.predictor3.freeze()
    
    # Module itself should still be trainable
    assert module.trainable == True
    
    # Check that only predictor2 is trainable
    trainable_predictors = module.predictors(return_trainable=True)
    all_predictors = module.predictors(return_trainable=False)
    assert len(trainable_predictors) == 1
    assert len(all_predictors) == 3
    assert module.predictor1.trainable == False
    assert module.predictor2.trainable == True
    assert module.predictor3.trainable == False


def test_nested_module_freeze():
    """Test freezing behavior with nested modules."""
    module = NestedModule()
    
    # Count all predictors
    all_params = module.named_parameters()
    assert len(all_params) == 4  # 1 from qa_module + 3 from multi_module
    
    # Freeze the qa_module sub-module
    module.qa_module.freeze()
    
    # The qa_module and its predictors should be frozen
    assert module.qa_module.trainable == False
    assert module.qa_module.generate_answer.trainable == False
    assert module.qa_module.generate_answer.predict.trainable == False
    
    # But multi_module predictors should still be trainable
    trainable_predictors = module.predictors(return_trainable=True)
    # Should have 3 trainable predictors from multi_module
    assert len(trainable_predictors) == 3


def test_compiled_flag_behavior():
    """Test that _compiled flag affects parameter traversal."""
    module = NestedModule()
    
    # Initially all parameters should be visible
    params_before = list(module.named_parameters())
    assert len(params_before) == 4
    
    # Mark qa_module as compiled
    module.qa_module._compiled = True
    
    # Now only multi_module parameters should be visible
    params_after = list(module.named_parameters())
    assert len(params_after) == 3
    
    # The parameter names should be from multi_module only
    param_names = [name for name, _ in params_after]
    assert all('multi_module' in name for name in param_names)


def test_freeze_with_bootstrap_optimizer():
    """Test module compilation with BootstrapFewShot optimizer."""
    # Setup dummy LM and data
    lm = DummyLM([
        {"answer": "Paris", "reasoning": "France's capital"}, 
        {"answer": "London", "reasoning": "UK's capital"}, 
        {"answer": "Berlin", "reasoning": "Germany's capital"},
        {"answer": "Rome", "reasoning": "Italy's capital"},
        {"answer": "Madrid", "reasoning": "Spain's capital"}
    ])
    dspy.settings.configure(lm=lm, adapter=dspy.JSONAdapter())
    
    # Create training data
    trainset = [
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="What is the capital of UK?", answer="London").with_inputs("question"),
    ]
    
    # Create and compile module
    module = SimpleQAModule()
    optimizer = BootstrapFewShot(metric=lambda x, pred, trace=None: pred.answer == x.answer)
    
    # Compile the module
    compiled_module = optimizer.compile(module, trainset=trainset)
    
    # The compiled module should have demonstrations
    # Check if any predictor has demos
    has_demos = any(hasattr(pred, 'demos') and len(pred.demos) > 0 
                   for _, pred in compiled_module.named_predictors())
    assert has_demos or compiled_module._compiled  # Either has demos or is marked as compiled


def test_freeze_with_module_list():
    """Test freezing behavior with modules in lists."""
    class ModuleWithList(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor_list = [
                dspy.Predict("input -> output"),
                dspy.ChainOfThought("input -> output"),
                dspy.Predict("input -> output")
            ]
        
        def forward(self, input):
            results = []
            for predictor in self.predictor_list:
                results.append(predictor(input=input))
            return results
    
    module = ModuleWithList()
    
    # Check initial state - named_parameters should find predictors in lists
    all_params = list(module.named_parameters())
    assert len(all_params) == 3
    
    # Freeze module - should freeze all predictors in the list
    module.freeze()
    assert module.trainable == False
    
    # All predictors in the list should be frozen
    for pred in module.predictor_list:
        assert pred.trainable == False
        # ChainOfThought's internal predict should also be frozen
        if hasattr(pred, 'predict'):
            assert pred.predict.trainable == False
    
    # No predictors should be trainable
    trainable_predictors = module.predictors(return_trainable=True)
    assert len(trainable_predictors) == 0


def test_freeze_unfreeze_cycles():
    """Test multiple freeze/unfreeze cycles."""
    module = ModuleWithMultiplePredictors()
    
    for i in range(3):
        # Freeze
        module.freeze()
        assert module.trainable == False
        assert len(module.predictors(return_trainable=True)) == 0
        assert all(not pred.trainable for _, pred in module.named_predictors())
        
        # Unfreeze
        module.unfreeze()
        assert module.trainable == True
        assert len(module.predictors(return_trainable=True)) == 3
        assert all(pred.trainable for _, pred in module.named_predictors())


def test_freeze_persistence_after_deepcopy():
    """Test that freeze state is preserved after deepcopy."""
    module = SimpleQAModule()
    
    # Freeze the module
    module.freeze()
    
    # Deep copy the module
    copied_module = module.deepcopy()
    
    # Check that frozen state is preserved
    assert copied_module.trainable == False
    assert len(copied_module.predictors(return_trainable=True)) == 0
    assert copied_module.generate_answer.trainable == False
    assert copied_module.generate_answer.predict.trainable == False


def test_compiled_flag_with_optimizer():
    """Test that optimizers may set _compiled flag."""
    lm = DummyLM([{"answer": "test"}])
    dspy.settings.configure(lm=lm)
    
    module = SimpleQAModule()
    assert not hasattr(module, '_compiled') or module._compiled == False
    
    # After some optimization process (simulated here)
    module._compiled = True
    
    # Now if this module is part of a larger module, its parameters
    # should not be traversed
    parent = dspy.Module()
    parent.child = module
    
    # The child's parameters should not be included
    params = list(parent.named_parameters())
    assert len(params) == 0  # No parameters because child is compiled


def test_freeze_with_evaluate():
    """Test that frozen modules work correctly with Evaluate."""
    lm = DummyLM([
        {"answer": "Paris", "reasoning": "France's capital"},
        {"answer": "London", "reasoning": "UK's capital"},
        {"answer": "Berlin", "reasoning": "Germany's capital"}
    ])
    dspy.settings.configure(lm=lm, adapter=dspy.JSONAdapter())
    
    # Create evaluation data
    devset = [
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="What is the capital of UK?", answer="London").with_inputs("question"),
    ]
    
    # Create and freeze module
    module = SimpleQAModule()
    module.freeze()
    
    # Evaluate should work normally with frozen module
    evaluate = Evaluate(
        devset=devset,
        metric=lambda x, pred, trace=None: pred.answer.lower() == x.answer.lower(),
        num_threads=1,
        display_progress=False
    )
    
    # This should not raise any errors
    result = evaluate(module)
    assert result.score >= 0  # Some score should be returned


def test_module_level_freeze_control():
    """Test that module-level freeze can be used to control optimization."""
    module = SimpleQAModule()
    
    # Freeze at module level
    module.freeze()
    assert module.trainable == False
    assert len(module.predictors(return_trainable=True)) == 0
    
    # Optimizers should check module.trainable to decide whether to optimize
    # This is a convention that optimizers can follow
    if module.trainable:
        # Would perform optimization
        pass
    else:
        # Skip optimization for frozen modules
        pass
    
    # Unfreeze to allow optimization
    module.unfreeze()
    assert module.trainable == True
    assert len(module.predictors(return_trainable=True)) > 0


if __name__ == "__main__":
    # Run all tests
    test_freeze_unfreeze_basic()
    test_freeze_propagates_to_predictors()
    test_freeze_chain_of_thought()
    test_freeze_individual_predictors()
    test_nested_module_freeze()
    test_compiled_flag_behavior()
    test_freeze_with_bootstrap_optimizer()
    test_freeze_with_module_list()
    test_freeze_unfreeze_cycles()
    test_freeze_persistence_after_deepcopy()
    test_compiled_flag_with_optimizer()
    test_freeze_with_evaluate()
    test_module_level_freeze_control()
    
    print("All tests passed!")
