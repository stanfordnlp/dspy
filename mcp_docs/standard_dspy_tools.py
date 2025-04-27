
import dspy
import os
def calculate_sum(a: int, b: int) -> int:
   """Calculate the sum of two integers."""
   return a + b

def main():
   """Main entry point for the async application."""
   print("Starting MCP client initialization...")
   # Configure DSPy with LLM
   LLM = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
   dspy.configure(lm=LLM)
   

   print("\nCreating MCPTools instance...")
   tools = [calculate_sum]
   
   # Create ReAct agent in the same async context
   react_agent = dspy.ReAct("input->output", tools)
   
   # Run the agent (will use the existing event loop)
   print("\nRunning ReAct agent...")
   react_result = react_agent(
      input="what is the sum of 5 and 10?",
   )
   
   
   print("\nReAct Result:")
   print(react_result)

if __name__ == "__main__":
   main()


