#!/usr/bin/env python3
"""Debug script to inspect DSPy ReAct predictions."""

import sys
from pathlib import Path

# Add parent directory to path to import dspy
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from swebench_adapter import SWeBenchAdapter
from swebench_tools import get_swebench_tools

def debug_prediction():
    """Debug a single prediction to see the structure."""
    
    # Set up a simple test case
    adapter = SWeBenchAdapter()
    
    # Create a mock instance
    mock_instance = {
        "instance_id": "test_debug",
        "problem_statement": "Fix the import error in the code",
        "repo": "test/repo"
    }
    
    # Create ReAct agent with tools
    tools = get_swebench_tools(work_dir=".")
    signature = adapter.create_react_signature()
    
    # Initialize a simple LM for testing (you'll need to set API key)
    lm = dspy.LM(model="gpt-3.5-turbo", api_key="your-api-key-here")
    dspy.settings.configure(lm=lm)
    
    react_agent = dspy.ReAct(signature=signature, tools=tools, max_iters=3)
    
    # Format input
    react_input = adapter.format_input_for_react(mock_instance)
    
    print("Input to ReAct:")
    print(react_input)
    print("\n" + "="*50 + "\n")
    
    # Run ReAct (commented out - uncomment if you have API key)
    # prediction = react_agent(**react_input)
    
    # Print prediction structure
    # print("Prediction structure:")
    # print(f"Type: {type(prediction)}")
    # print(f"Attributes: {dir(prediction)}")
    # 
    # if hasattr(prediction, 'trajectory'):
    #     print(f"Trajectory keys: {list(prediction.trajectory.keys()) if prediction.trajectory else 'None'}")
    # 
    # if hasattr(prediction, 'solution'):
    #     print(f"Solution: {prediction.solution[:200]}...")
    # 
    # # Test patch extraction
    # patch = adapter._extract_patch_from_prediction(prediction, ".")
    # print(f"Extracted patch: {patch[:200]}...")

if __name__ == "__main__":
    debug_prediction()