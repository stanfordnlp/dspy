# Building Memory-Enabled Agents with DSPy ReAct and Mem0

This tutorial demonstrates how to build intelligent conversational agents that can remember information across interactions using DSPy's ReAct framework combined with [Mem0](https://docs.mem0.ai/)'s memory capabilities. You'll learn to create agents that can store, retrieve, and use contextual information to provide personalized and coherent responses.

## What You'll Build

By the end of this tutorial, you'll have a memory-enabled agent that can:

- **Remember user preferences** and past conversations
- **Store and retrieve factual information** about users and topics
- **Use memory to inform decisions** and provide personalized responses
- **Handle complex multi-turn conversations** with context awareness
- **Manage different types of memories** (facts, preferences, experiences)

## Prerequisites

- Basic understanding of DSPy and ReAct agents
- Python 3.9+ installed
- API keys for your preferred LLM provider

## Installation and Setup

```bash
pip install dspy mem0ai
```

## Step 1: Understanding Mem0 Integration

Mem0 provides a memory layer that can store, search, and retrieve memories for AI agents. Let's start by understanding how to integrate it with DSPy:

```python
import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure environment
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize Mem0 memory system
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
```

## Step 2: Create Memory-Aware Tools

Let's create tools that can interact with the memory system:

```python
import datetime

class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """Store information in memory."""
        try:
            self.memory.add(content, user_id=user_id)
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """Search for relevant memories."""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            if not results:
                return "No relevant memories found."

            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results:
                return "No memories found for this user."

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            self.memory.update(memory_id, new_content)
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## Step 3: Build the Memory-Enhanced ReAct Agent

Now let's create our main ReAct agent that can use memory:

```python
class MemoryQA(dspy.Signature):
    """
    You're a helpful assistant and have access to memory method.
    Whenever you answer a user's input, remember to store the information in memory
    so that you can use it later.
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class MemoryReActAgent(dspy.Module):
    """A ReAct agent enhanced with Mem0 memory capabilities."""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # Create tools list for ReAct
        self.tools = [
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # Initialize ReAct with our tools
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """Process user input with memory-aware reasoning."""
        
        return self.react(user_input=user_input)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """Set a reminder for the user."""
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        )

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """Get user preferences for a specific category."""
        query = f"user preferences {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """Update user preferences."""
        preference_text = f"User preference for {category}: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
```

## Step 4: Running the Memory-Enhanced Agent

Let's create a simple interface to interact with our memory-enabled agent:

```python
import time
def run_memory_agent_demo():
    """Demonstration of memory-enhanced ReAct agent."""

    # Configure DSPy
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Initialize memory system
    memory = Memory.from_config(config)

    # Create our agent
    agent = MemoryReActAgent(memory)

    # Sample conversation demonstrating memory capabilities
    print("🧠 Memory-Enhanced ReAct Agent Demo")
    print("=" * 50)

    conversations = [
        "Hi, I'm Alice and I love Italian food, especially pasta carbonara.",
        "I'm Alice. I prefer to exercise in the morning around 7 AM.",
        "I'm Alice. What do you remember about my food preferences?",
        "I'm Alice. Set a reminder for me to go grocery shopping tomorrow.",
        "I'm Alice. What are my exercise preferences?",
        "I'm Alice. I also enjoy hiking on weekends.",
        "I'm Alice. What do you know about me so far?"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 User: {user_input}")

        try:
            response = agent(user_input=user_input)
            print(f"🤖 Agent: {response.response}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")

# Run the demonstration
if __name__ == "__main__":
    run_memory_agent_demo()
```

## Expected Output
```
🧠 Memory-Enhanced ReAct Agent Demo
==================================================

📝 User: Hi, I'm Alice and I love Italian food, especially pasta carbonara.

🤖 Agent: Hi Alice! It's great to meet you! Pasta carbonara is a delicious choice. Do you have a favorite place to get it, or do you enjoy making it at home?

📝 User: I'm Alice. I prefer to exercise in the morning around 7 AM.

🤖 Agent: Nice to meet you, Alice! I've noted that you prefer to exercise in the morning around 7 AM. If you have any questions or need tips related to your morning workouts, feel free to ask!

📝 User: I'm Alice. What do you remember about my food preferences?

🤖 Agent: I remember that you love Italian food, and your favorite dish is pasta carbonara!

📝 User: I'm Alice. Set a reminder for me to go grocery shopping tomorrow.

🤖 Agent: I have set a reminder for you, Alice, to go grocery shopping tomorrow.

📝 User: I'm Alice. What are my exercise preferences?

🤖 Agent: Your exercise preference is to work out in the morning around 7 AM.

📝 User: I'm Alice. I also enjoy hiking on weekends.

🤖 Agent: Nice to meet you, Alice! It's great to hear that you enjoy hiking on weekends. If you need any recommendations for trails or tips for your hikes, feel free to ask!

📝 User: I'm Alice. What do you know about me so far?

🤖 Agent: So far, I know that you enjoy hiking on weekends, prefer to exercise in the morning around 7 AM, and love Italian food, especially pasta carbonara. Additionally, you have a reminder set for grocery shopping on October 7, 2023.
```

## Next Steps

- **Implement memory persistence** with databases (PostgreSQL, MongoDB)
- **Add memory categorization** and tagging for better organization
- **Create memory expiration policies** for data management
- **Build multi-user memory isolation** for production applications
- **Add memory analytics** and insights
- **Integrate with vector databases** for enhanced semantic search
- **Implement memory compression** for long-term storage efficiency

This tutorial demonstrates how DSPy's ReAct framework can be enhanced with Mem0's memory capabilities to create intelligent, context-aware agents that can learn and remember information across interactions, making them more useful for real-world applications.