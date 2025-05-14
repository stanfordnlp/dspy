import dspy
from dspy.propose.instruction_proposal import GenerateInstructionGivenAttempts
from dspy.utils.dummies import DummyLM


def test_generate_instruction_given_attempts_minimal():
    """Test that the GenerateInstructionGivenAttempts signature initializes correctly"""
    
    dummy_lm = DummyLM([{
        "proposed_instruction": "Better instruction",
        "proposed_prefix_for_output_field": "Result:",
    }])
    dspy.settings.configure(lm=dummy_lm)
    
    # Create a predictor with the signature
    predictor = dspy.Predict(GenerateInstructionGivenAttempts)
    
    # Basic test - just ensure we can call it without errors
    result = predictor(attempted_instructions=["Instruction #1: Test", "Score #1: 0.5"])
    
    # Verify structure of the result
    assert hasattr(result, "proposed_instruction")
    assert hasattr(result, "proposed_prefix_for_output_field")