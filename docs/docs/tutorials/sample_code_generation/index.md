# Automated Code Generation from Documentation with DSPy

This tutorial demonstrates how to use DSPy to automatically fetch documentation from URLs and generate working code examples for any library. The system can analyze documentation websites, extract key concepts, and produce tailored code examples.

## What You'll Build

A documentation-powered code generation system that:

- Fetches and parses documentation from multiple URLs
- Extracts API patterns, methods, and usage examples  
- Generates working code for specific use cases
- Provides explanations and best practices
- Works with any library's documentation

## Setup

```bash
pip install dspy requests beautifulsoup4 html2text
```

## Step 1: Documentation Fetching and Processing

```python
import dspy
import requests
from bs4 import BeautifulSoup
import html2text
from typing import List, Dict, Any
import json
from urllib.parse import urljoin, urlparse
import time

# Configure DSPy
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

class DocumentationFetcher:
    """Fetches and processes documentation from URLs."""
    
    def __init__(self, max_retries=3, delay=1):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.max_retries = max_retries
        self.delay = delay
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
    
    def fetch_url(self, url: str) -> Dict[str, str]:
        """Fetch content from a single URL."""
        for attempt in range(self.max_retries):
            try:
                print(f"📡 Fetching: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Convert to markdown for better LLM processing
                markdown_content = self.html_converter.handle(str(soup))
                
                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "No title",
                    "content": markdown_content,
                    "success": True
                }
                
            except Exception as e:
                print(f"❌ Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay)
                else:
                    return {
                        "url": url,
                        "title": "Failed to fetch",
                        "content": f"Error: {str(e)}",
                        "success": False
                    }
        
        return {"url": url, "title": "Failed", "content": "", "success": False}
    
    def fetch_documentation(self, urls: List[str]) -> List[Dict[str, str]]:
        """Fetch documentation from multiple URLs."""
        results = []
        
        for url in urls:
            result = self.fetch_url(url)
            results.append(result)
            time.sleep(self.delay)  # Be respectful to servers
        
        return results

class LibraryAnalyzer(dspy.Signature):
    """Analyze library documentation to understand core concepts and patterns."""
    library_name: str = dspy.InputField(desc="Name of the library to analyze")
    documentation_content: str = dspy.InputField(desc="Combined documentation content")
    
    core_concepts: List[str] = dspy.OutputField(desc="Main concepts and components")
    common_patterns: List[str] = dspy.OutputField(desc="Common usage patterns")
    key_methods: List[str] = dspy.OutputField(desc="Important methods and functions")
    installation_info: str = dspy.OutputField(desc="Installation and setup information")
    code_examples: List[str] = dspy.OutputField(desc="Example code snippets found")

class CodeGenerator(dspy.Signature):
    """Generate code examples for specific use cases using the target library."""
    library_info: str = dspy.InputField(desc="Library concepts and patterns")
    use_case: str = dspy.InputField(desc="Specific use case to implement")
    requirements: str = dspy.InputField(desc="Additional requirements or constraints")
    
    code_example: str = dspy.OutputField(desc="Complete, working code example")
    explanation: str = dspy.OutputField(desc="Step-by-step explanation of the code")
    best_practices: List[str] = dspy.OutputField(desc="Best practices and tips")
    imports_needed: List[str] = dspy.OutputField(desc="Required imports and dependencies")

class DocumentationLearningAgent(dspy.Module):
    """Agent that learns from documentation URLs and generates code examples."""
    
    def __init__(self):
        super().__init__()
        self.fetcher = DocumentationFetcher()
        self.analyze_docs = dspy.ChainOfThought(LibraryAnalyzer)
        self.generate_code = dspy.ChainOfThought(CodeGenerator)
        self.refine_code = dspy.ChainOfThought(
            "code, feedback -> improved_code: str, changes_made: List[str]"
        )
    
    def learn_from_urls(self, library_name: str, doc_urls: List[str]) -> Dict:
        """Learn about a library from its documentation URLs."""
        
        print(f"📚 Learning about {library_name} from {len(doc_urls)} URLs...")
        
        # Fetch all documentation
        docs = self.fetcher.fetch_documentation(doc_urls)
        
        # Combine successful fetches
        combined_content = "\n\n---\n\n".join([
            f"URL: {doc['url']}\nTitle: {doc['title']}\n\n{doc['content']}"
            for doc in docs if doc['success']
        ])
        
        if not combined_content:
            raise ValueError("No documentation could be fetched successfully")
        
        # Analyze combined documentation
        analysis = self.analyze_docs(
            library_name=library_name,
            documentation_content=combined_content
        )
        
        return {
            "library": library_name,
            "source_urls": [doc['url'] for doc in docs if doc['success']],
            "core_concepts": analysis.core_concepts,
            "patterns": analysis.common_patterns,
            "methods": analysis.key_methods,
            "installation": analysis.installation_info,
            "examples": analysis.code_examples,
            "fetched_docs": docs
        }
    
    def generate_example(self, library_info: Dict, use_case: str, requirements: str = "") -> Dict:
        """Generate a code example for a specific use case."""
        
        # Format library information for the generator
        info_text = f"""
        Library: {library_info['library']}
        Core Concepts: {', '.join(library_info['core_concepts'])}
        Common Patterns: {', '.join(library_info['patterns'])}
        Key Methods: {', '.join(library_info['methods'])}
        Installation: {library_info['installation']}
        Example Code Snippets: {'; '.join(library_info['examples'][:3])}  # First 3 examples
        """
        
        code_result = self.generate_code(
            library_info=info_text,
            use_case=use_case,
            requirements=requirements
        )
        
        return {
            "code": code_result.code_example,
            "explanation": code_result.explanation,
            "best_practices": code_result.best_practices,
            "imports": code_result.imports_needed
        }

# Initialize the learning agent
agent = DocumentationLearningAgent()
```

## Step 2: Learning from Documentation URLs

```python
def learn_library_from_urls(library_name: str, documentation_urls: List[str]) -> Dict:
    """Learn about any library from its documentation URLs."""
    
    try:
        library_info = agent.learn_from_urls(library_name, documentation_urls)
        
        print(f"\n🔍 Library Analysis Results for {library_name}:")
        print(f"Sources: {len(library_info['source_urls'])} successful fetches")
        print(f"Core Concepts: {library_info['core_concepts']}")
        print(f"Common Patterns: {library_info['patterns']}")
        print(f"Key Methods: {library_info['methods']}")
        print(f"Installation: {library_info['installation']}")
        print(f"Found {len(library_info['examples'])} code examples")
        
        return library_info
        
    except Exception as e:
        print(f"❌ Error learning library: {e}")
        raise

# Example 1: Learn FastAPI from official documentation
fastapi_urls = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/"
]

print("🚀 Learning FastAPI from official documentation...")
fastapi_info = learn_library_from_urls("FastAPI", fastapi_urls)

# Example 2: Learn a different library (you can replace with any library)
streamlit_urls = [
    "https://docs.streamlit.io/",
    "https://docs.streamlit.io/get-started",
    "https://docs.streamlit.io/develop/api-reference"
]

print("\n\n📊 Learning Streamlit from official documentation...")
streamlit_info = learn_library_from_urls("Streamlit", streamlit_urls)
```

## Step 3: Generating Code Examples

```python
def generate_examples_for_library(library_info: Dict, library_name: str):
    """Generate code examples for any library based on its documentation."""
    
    # Define generic use cases that can apply to most libraries
    use_cases = [
        {
            "name": "Basic Setup and Hello World",
            "description": f"Create a minimal working example with {library_name}",
            "requirements": "Include installation, imports, and basic usage"
        },
        {
            "name": "Common Operations",
            "description": f"Demonstrate the most common {library_name} operations",
            "requirements": "Show typical workflow and best practices"
        },
        {
            "name": "Advanced Usage",
            "description": f"Create a more complex example showcasing {library_name} capabilities",
            "requirements": "Include error handling and optimization"
        }
    ]
    
    generated_examples = []
    
    print(f"\n🔧 Generating examples for {library_name}...")
    
    for use_case in use_cases:
        print(f"\n📝 {use_case['name']}")
        print(f"Description: {use_case['description']}")
        
        example = agent.generate_example(
            library_info=library_info,
            use_case=use_case['description'],
            requirements=use_case['requirements']
        )
        
        print("\n💻 Generated Code:")
        print("```python")
        print(example['code'])
        print("```")
        
        print("\n📦 Required Imports:")
        for imp in example['imports']:
            print(f"  • {imp}")
        
        print("\n📝 Explanation:")
        print(example['explanation'])
        
        print("\n✅ Best Practices:")
        for practice in example['best_practices']:
            print(f"  • {practice}")
        
        generated_examples.append({
            "use_case": use_case['name'],
            "code": example['code'],
            "imports": example['imports'],
            "explanation": example['explanation'],
            "best_practices": example['best_practices']
        })
        
        print("-" * 80)
    
    return generated_examples

# Generate examples for both libraries
print("🎯 Generating FastAPI Examples:")
fastapi_examples = generate_examples_for_library(fastapi_info, "FastAPI")

print("\n\n🎯 Generating Streamlit Examples:")
streamlit_examples = generate_examples_for_library(streamlit_info, "Streamlit")
```

## Step 4: Generic Library Learning Function

```python
def learn_any_library(library_name: str, documentation_urls: List[str], use_cases: List[str] = None):
    """Learn any library from its documentation and generate examples."""
    
    if use_cases is None:
        use_cases = [
            "Basic setup and hello world example",
            "Common operations and workflows",
            "Advanced usage with best practices"
        ]
    
    print(f"🚀 Starting automated learning for {library_name}...")
    print(f"Documentation sources: {len(documentation_urls)} URLs")
    
    try:
        # Step 1: Learn from documentation
        library_info = agent.learn_from_urls(library_name, documentation_urls)
        
        # Step 2: Generate examples for each use case
        all_examples = []
        
        for i, use_case in enumerate(use_cases, 1):
            print(f"\n📝 Generating example {i}/{len(use_cases)}: {use_case}")
            
            example = agent.generate_example(
                library_info=library_info,
                use_case=use_case,
                requirements="Include error handling, comments, and follow best practices"
            )
            
            all_examples.append({
                "use_case": use_case,
                "code": example['code'],
                "imports": example['imports'],
                "explanation": example['explanation'],
                "best_practices": example['best_practices']
            })
        
        return {
            "library_info": library_info,
            "examples": all_examples
        }
    
    except Exception as e:
        print(f"❌ Error learning {library_name}: {e}")
        return None

# Example usage with different libraries
libraries_to_learn = [
    {
        "name": "Plotly",
        "urls": [
            "https://plotly.com/python/getting-started/",
            "https://plotly.com/python/basic-charts/"
        ],
        "use_cases": [
            "Create a simple line chart",
            "Build an interactive dashboard",
            "Advanced styling and customization"
        ]
    },
    {
        "name": "Rich",
        "urls": [
            "https://rich.readthedocs.io/en/stable/introduction.html",
            "https://rich.readthedocs.io/en/stable/console.html",
            "https://rich.readthedocs.io/en/stable/text.html"
        ],
        "use_cases": [
            "Basic console output with styling",
            "Create progress bars and tables",
            "Build complex terminal interfaces"
        ]
    }
]

# Learn multiple libraries
learned_libraries = {}

for lib_config in libraries_to_learn:
    print(f"\n{'='*60}")
    print(f"Learning {lib_config['name']}...")
    print(f"{'='*60}")
    
    result = learn_any_library(
        library_name=lib_config['name'],
        documentation_urls=lib_config['urls'],
        use_cases=lib_config['use_cases']
    )
    
    if result:
        learned_libraries[lib_config['name']] = result
        print(f"✅ Successfully learned {lib_config['name']}!")
    else:
        print(f"❌ Failed to learn {lib_config['name']}")
```

## Example Output

When you run the automated learning system, you'll see:

**Documentation Fetching:**
```
🚀 Starting automated learning for FastAPI...
Documentation sources: 4 URLs
📡 Fetching: https://fastapi.tiangolo.com/ (attempt 1)
📡 Fetching: https://fastapi.tiangolo.com/tutorial/first-steps/ (attempt 1)
📡 Fetching: https://fastapi.tiangolo.com/tutorial/path-params/ (attempt 1)
📚 Learning about FastAPI from 4 URLs...

🔍 Library Analysis Results for FastAPI:
Sources: 4 successful fetches
Core Concepts: ['FastAPI app', 'path operations', 'dependencies', 'request/response models']
Common Patterns: ['app = FastAPI()', 'decorator-based routing', 'Pydantic models']
Key Methods: ['FastAPI()', '@app.get()', '@app.post()', 'uvicorn.run()']
Installation: pip install fastapi uvicorn
```

**Generated Code Example:**
```python
📝 Generating example 1/3: Basic setup and hello world example

💻 Generated Code:
from fastapi import FastAPI
import uvicorn
from typing import Dict, Optional

# Create FastAPI instance
app = FastAPI(
    title="My API",
    description="A simple FastAPI example",
    version="1.0.0"
)

@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Hello World from FastAPI!"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

📦 Required Imports:
  • pip install fastapi uvicorn
  • from fastapi import FastAPI
  • import uvicorn

📝 Explanation:
This example creates a basic FastAPI application with two endpoints...

✅ Best Practices:
  • Use async/await for better performance
  • Include proper type hints
  • Add documentation metadata to FastAPI instance
```


## Next Steps

- **GitHub Integration**: Learn from README files and example repositories
- **Video Tutorial Processing**: Extract information from video documentation
- **Community Examples**: Aggregate examples from Stack Overflow and forums
- **Version Comparison**: Track API changes across library versions
- **Testing Generation**: Automatically create unit tests for generated code

This tutorial demonstrates how DSPy can automate the entire process of learning unfamiliar libraries from their documentation, making it valuable for rapid technology adoption and exploration.
