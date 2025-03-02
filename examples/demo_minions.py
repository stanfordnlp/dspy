#!/usr/bin/env python3
"""
Demo script for MinionsLM and StructuredMinionsLM in the DSPy framework.

This script demonstrates the usage of MinionsLM and StructuredMinionsLM by:
1. Initializing local and remote language models
2. Creating instances of MinionsLM and StructuredMinionsLM
3. Running various queries to demonstrate their capabilities
4. Displaying performance metrics
"""

import time
import argparse
from typing import List

import dspy
from dspy.clients import MinionsLM, create_minions_lm
from dspy.minions import StructuredMinionsLM, create_structured_minions


def print_divider(title: str) -> None:
    """
    Print a divider with a title.
    
    Args:
        title: The title to display in the divider
    """
    line_length = 60
    title_length = len(title)
    padding = (line_length - title_length - 2) // 2
    print("\n" + "=" * line_length)
    print(" " * padding + f" {title} " + " " * padding)
    print("=" * line_length + "\n")


class TestLocalLM(dspy.LM):
    """
    A test local language model for simulating responses when real models aren't available.
    """
    def __init__(self) -> None:
        super().__init__(model="test-local")
        self.name = "TestLocalLM"
    
    def __call__(self, prompt: str, **kwargs) -> List[str]:
        """Simulate a response to a prompt"""
        if "primary colors" in prompt.lower():
            return ["The primary colors are red, blue, and yellow."]
        elif "capitals" in prompt.lower():
            return ["Washington D.C. is the capital of the United States."]
        elif "recipe" in prompt.lower():
            return ["""
            {
                "title": "Simple Pasta",
                "ingredients": ["200g pasta", "100g cheese", "1 tbsp olive oil", "salt to taste"],
                "instructions": ["Boil water", "Add pasta and cook", "Drain and add cheese", "Season with salt"],
                "prep_time": "5 minutes",
                "cook_time": "10 minutes"
            }
            """]
        else:
            return ["This is a simulated response from the local language model."]


class TestRemoteLM(dspy.LM):
    """
    A test remote language model for simulating responses when real models aren't available.
    """
    def __init__(self) -> None:
        super().__init__(model="test-remote")
        self.name = "TestRemoteLM"
    
    def __call__(self, prompt: str, **kwargs) -> List[str]:
        """Simulate a response to a prompt"""
        if "document" in prompt.lower():
            return ["""
            {
                "summary": "This is a summary of the document that discusses climate change impact on global economies.",
                "key_points": ["Rising temperatures affect agriculture", "Coastal cities face flooding risks", "Economic impacts vary by region"],
                "topics": ["Climate Change", "Economics", "Adaptation Strategies"],
                "sentiment": "concerned",
                "entities": ["IPCC", "World Bank", "United Nations"]
            }
            """]
        elif "compare" in prompt.lower():
            return ["""
            {
                "similarities": ["Both are programming languages", "Both support object-oriented programming"],
                "differences": ["Python is interpreted, Java is compiled", "Python uses indentation, Java uses braces"],
                "pros_entity1": ["Easy to learn", "Large ecosystem"],
                "pros_entity2": ["Strong type system", "Performance"],
                "cons_entity1": ["Slower execution", "Global interpreter lock"],
                "cons_entity2": ["More verbose", "Steeper learning curve"],
                "recommendation": "Python for rapid development, Java for enterprise applications"
            }
            """]
        else:
            return ["This is a detailed simulated response from the remote language model that goes into much more depth and complexity than the local model would provide."]


def setup_models(local_model_name: str = "ollama/llama3.2", remote_model_name: str = "gpt-3.5-turbo") -> tuple:
    """
    Set up local and remote language models for the demo.
    
    Args:
        local_model_name: Name of the local model to use
        remote_model_name: Name of the remote model to use
        
    Returns:
        Tuple containing (local_lm, remote_lm)
    """
    # Try to initialize real models
    try:
        print(f"Initializing local model: {local_model_name}")
        # Use LM class to avoid specific model implementation requirements
        if "ollama" in local_model_name.lower():
            # If ollama-specific class exists
            try:
                from dspy.clients.lm_local import OllamaLM
                local_lm = OllamaLM(model=local_model_name)
            except (ImportError, AttributeError):
                # Fall back to generic LM
                local_lm = dspy.LM(model=local_model_name)
        else:
            local_lm = dspy.LM(model=local_model_name)
    except Exception as e:
        print(f"Failed to initialize local model: {e}")
        print("Using simulated local model instead")
        local_lm = TestLocalLM()
    
    try:
        print(f"Initializing remote model: {remote_model_name}")
        # Try to use OpenAI client if available
        if "gpt" in remote_model_name.lower():
            try:
                from dspy.clients.openai import OpenAI
                remote_lm = OpenAI(model=remote_model_name)
            except (ImportError, AttributeError):
                # Fall back to generic LM
                remote_lm = dspy.LM(model=remote_model_name)
        else:
            remote_lm = dspy.LM(model=remote_model_name)
    except Exception as e:
        print(f"Failed to initialize remote model: {e}")
        print("Using simulated remote model instead")
        remote_lm = TestRemoteLM()
    
    return local_lm, remote_lm


def test_simple_qa(minions_lm: MinionsLM) -> None:
    """
    Test the MinionsLM with simple question answering.
    
    Args:
        minions_lm: The MinionsLM instance to test
    """
    print_divider("Testing Simple Questions with MinionsLM")
    
    # Simple question that should be handled by the local model
    question1 = "What are the primary colors?"
    print(f"Question: {question1}")
    start_time = time.time()
    response1 = minions_lm(prompt=question1)
    elapsed1 = time.time() - start_time
    print(f"Response: {response1[0]}")
    print(f"Time: {elapsed1:.2f}s")
    print(f"Routed to: {'local' if minions_lm.metrics['local_calls'] > 0 else 'remote'} model")
    print(f"Local calls: {minions_lm.metrics['local_calls']}")
    print(f"Remote calls: {minions_lm.metrics['remote_calls']}")
    
    # Reset metrics
    minions_lm.reset_metrics()
    
    # More complex question that might be routed to the remote model
    question2 = "Explain the implications of quantum computing on modern cryptography and how organizations should prepare for quantum threats to their security infrastructure."
    print(f"\nQuestion: {question2}")
    start_time = time.time()
    response2 = minions_lm(prompt=question2)
    elapsed2 = time.time() - start_time
    print(f"Response: {response2[0][:200]}...")  # Truncated for brevity
    print(f"Time: {elapsed2:.2f}s")
    print(f"Routed to: {'local' if minions_lm.metrics['local_calls'] > 0 else 'remote'} model")
    print(f"Local calls: {minions_lm.metrics['local_calls']}")
    print(f"Remote calls: {minions_lm.metrics['remote_calls']}")


def test_structured_qa(structured_minions: StructuredMinionsLM) -> None:
    """
    Test the StructuredMinionsLM with structured question answering.
    
    Args:
        structured_minions: The StructuredMinionsLM instance to test
    """
    print_divider("Testing Structured QA with StructuredMinionsLM")
    
    question = "What are the capitals of the G7 countries?"
    print(f"Question: {question}")
    
    start_time = time.time()
    response = structured_minions.process_query(question)
    elapsed = time.time() - start_time
    
    print("Structured Response:")
    for key, value in response.items():
        print(f"  {key}: {value}")
    
    print(f"\nTime: {elapsed:.2f}s")
    metrics = structured_minions.get_metrics()
    print(f"Local calls: {metrics['local_calls']}")
    print(f"Remote calls: {metrics['remote_calls']}")


def test_document_analysis(structured_minions: StructuredMinionsLM) -> None:
    """
    Test the StructuredMinionsLM with document analysis.
    
    Args:
        structured_minions: The StructuredMinionsLM instance to test
    """
    print_divider("Testing Document Analysis with StructuredMinionsLM")
    
    document = """
    Climate Change Impact Report - 2024
    
    Global temperatures have increased by 1.1°C since pre-industrial times, with the last decade being the warmest on record. This warming has led to more frequent extreme weather events, including heatwaves, droughts, and floods. The economic impact of these changes is estimated at $500 billion annually.
    
    Key findings:
    1. Arctic sea ice is declining at a rate of 13% per decade
    2. Global sea levels are rising at 3.3mm per year
    3. Ocean acidification has increased by 30% since the industrial revolution
    
    Mitigation strategies include:
    - Rapid transition to renewable energy
    - Carbon capture and storage technologies
    - Sustainable agriculture and forestry practices
    - International cooperation through agreements like the Paris Accord
    
    Adaptation measures are necessary, especially for vulnerable communities in low-lying coastal areas and regions prone to extreme weather events.
    """
    
    print(f"Document length: {len(document)} characters")
    
    start_time = time.time()
    analysis = structured_minions.process_document(document)
    elapsed = time.time() - start_time
    
    print("Document Analysis:")
    for key, value in analysis.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value[:3]:  # Show first 3 items
                print(f"    - {item}")
            if len(value) > 3:
                print(f"    - ... ({len(value) - 3} more items)")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTime: {elapsed:.2f}s")
    metrics = structured_minions.get_metrics()
    print(f"Local calls: {metrics['local_calls']}")
    print(f"Remote calls: {metrics['remote_calls']}")


def test_recipe_extraction(structured_minions: StructuredMinionsLM) -> None:
    """
    Test the StructuredMinionsLM with recipe extraction.
    
    Args:
        structured_minions: The StructuredMinionsLM instance to test
    """
    print_divider("Testing Recipe Extraction with StructuredMinionsLM")
    
    recipe_text = """
    Classic Chocolate Chip Cookies
    
    Ingredients:
    - 2 1/4 cups all-purpose flour
    - 1 teaspoon baking soda
    - 1 teaspoon salt
    - 1 cup (2 sticks) unsalted butter, softened
    - 3/4 cup granulated sugar
    - 3/4 cup packed brown sugar
    - 2 large eggs
    - 2 teaspoons vanilla extract
    - 2 cups (12 oz) semi-sweet chocolate chips
    
    Instructions:
    1. Preheat oven to 375°F (190°C).
    2. Combine flour, baking soda, and salt in a small bowl.
    3. Beat butter, granulated sugar, and brown sugar in a large mixer bowl until creamy.
    4. Add eggs one at a time, beating well after each addition. Add vanilla.
    5. Gradually beat in flour mixture. Stir in chocolate chips.
    6. Drop by rounded tablespoon onto ungreased baking sheets.
    7. Bake for 9 to 11 minutes or until golden brown.
    8. Cool on baking sheets for 2 minutes; remove to wire racks to cool completely.
    
    Prep time: 15 minutes
    Cook time: 10 minutes
    Yield: About 5 dozen cookies
    """
    
    print(f"Recipe text length: {len(recipe_text)} characters")
    
    start_time = time.time()
    recipe_info = structured_minions.process_recipe(recipe_text)
    elapsed = time.time() - start_time
    
    print("Recipe Extraction:")
    for key, value in recipe_info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value[:3]:  # Show first 3 items
                print(f"    - {item}")
            if len(value) > 3:
                print(f"    - ... ({len(value) - 3} more items)")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTime: {elapsed:.2f}s")
    metrics = structured_minions.get_metrics()
    print(f"Local calls: {metrics['local_calls']}")
    print(f"Remote calls: {metrics['remote_calls']}")


def test_dspy_integration(minions_lm: MinionsLM) -> None:
    """
    Test integration with DSPy modules.
    
    Args:
        minions_lm: The MinionsLM instance to test
    """
    print_divider("Testing DSPy Integration")
    
    # Create a prompt for summarization
    climate_doc = """
    Climate change is the long-term alteration of temperature and typical weather patterns. 
    The cause of current climate change is largely human activity, like burning fossil fuels, 
    which adds heat-trapping gases to Earth's atmosphere. These activities increase atmospheric 
    carbon dioxide levels, creating a greenhouse effect that leads to global warming and climate change.
    
    The effects of climate change are far-reaching. Rising global temperatures lead to sea level rise, 
    more intense and frequent extreme weather events, and shifts in wildlife populations and habitats. 
    These changes affect food security, water supply, health, and infrastructure.
    
    Addressing climate change requires both mitigation and adaptation strategies. Mitigation involves 
    reducing or preventing greenhouse gas emissions, while adaptation involves adjusting to current or 
    expected effects of climate change. International cooperation, policy changes, technological innovations, 
    and individual actions all play important roles in combating climate change.
    """
    
    prompt = f"Please summarize the following document about climate change: {climate_doc}"
    
    print(f"Prompt: {prompt[:100]}...")
    
    start_time = time.time()
    result = minions_lm(prompt=prompt)
    elapsed = time.time() - start_time
    
    print(f"Summary: {result[0]}")
    print(f"Time: {elapsed:.2f}s")
    
    metrics = minions_lm.metrics
    print(f"Local calls: {metrics['local_calls']}")
    print(f"Remote calls: {metrics['remote_calls']}")


def print_metrics(minions_lm: MinionsLM) -> None:
    """
    Print metrics from the MinionsLM instance.
    
    Args:
        minions_lm: The MinionsLM instance to get metrics from
    """
    print_divider("Performance Metrics")
    
    metrics = minions_lm.metrics
    
    # Print metrics if available, or show defaults
    print(f"Local model calls: {metrics.get('local_calls', 0)}")
    print(f"Remote model calls: {metrics.get('remote_calls', 0)}")
    print(f"Average local call time: {metrics.get('avg_local_time', 0):.2f}s")
    print(f"Average remote call time: {metrics.get('avg_remote_time', 0):.2f}s")
    
    # Calculate total calls
    total_calls = metrics.get('local_calls', 0) + metrics.get('remote_calls', 0)
    print(f"Total calls: {total_calls}")
    
    # Calculate local vs remote ratio if there were any calls
    if total_calls > 0:
        local_ratio = metrics.get('local_calls', 0) / total_calls * 100
        print(f"Local routing ratio: {local_ratio:.1f}%")
    
    # Show other metrics if available
    for key in sorted(metrics.keys()):
        if key not in ['local_calls', 'remote_calls', 'avg_local_time', 'avg_remote_time']:
            print(f"{key}: {metrics[key]}")


def main() -> None:
    """
    Main function to run the demo.
    """
    parser = argparse.ArgumentParser(description="Demo script for MinionsLM and StructuredMinionsLM")
    parser.add_argument("--local-model", type=str, default="ollama/llama3.2", help="Local model name")
    parser.add_argument("--remote-model", type=str, default="gpt-3.5-turbo", help="Remote model name")
    parser.add_argument("--complexity-threshold", type=float, default=0.6, help="Complexity threshold (0.0-1.0)")
    parser.add_argument("--max-parallel", type=int, default=3, help="Maximum parallel chunks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set up the local and remote language models
    local_lm, remote_lm = setup_models(args.local_model, args.remote_model)
    
    # Create MinionsLM and StructuredMinionsLM instances
    minions_lm = create_minions_lm(
        local_lm=local_lm,
        remote_lm=remote_lm,
        complexity_threshold=args.complexity_threshold,
        max_parallel_chunks=args.max_parallel,
        verbose=args.verbose
    )
    
    structured_minions = create_structured_minions(
        local_lm=local_lm,
        remote_lm=remote_lm,
        complexity_threshold=args.complexity_threshold,
        max_parallel_chunks=args.max_parallel,
        verbose=args.verbose
    )
    
    # Run the demo
    test_simple_qa(minions_lm)
    test_structured_qa(structured_minions)
    test_document_analysis(structured_minions)
    test_recipe_extraction(structured_minions)
    test_dspy_integration(minions_lm)
    
    # Print final metrics
    print_metrics(minions_lm)


if __name__ == "__main__":
    main() 