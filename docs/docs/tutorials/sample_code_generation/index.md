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
    
    def fetch_url(self, url: str) -> dict[str, str]:
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
    
    def fetch_documentation(self, urls: list[str]) -> list[dict[str, str]]:
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
    
    core_concepts: list[str] = dspy.OutputField(desc="Main concepts and components")
    common_patterns: list[str] = dspy.OutputField(desc="Common usage patterns")
    key_methods: list[str] = dspy.OutputField(desc="Important methods and functions")
    installation_info: str = dspy.OutputField(desc="Installation and setup information")
    code_examples: list[str] = dspy.OutputField(desc="Example code snippets found")

class CodeGenerator(dspy.Signature):
    """Generate code examples for specific use cases using the target library."""
    library_info: str = dspy.InputField(desc="Library concepts and patterns")
    use_case: str = dspy.InputField(desc="Specific use case to implement")
    requirements: str = dspy.InputField(desc="Additional requirements or constraints")
    
    code_example: str = dspy.OutputField(desc="Complete, working code example")
    explanation: str = dspy.OutputField(desc="Step-by-step explanation of the code")
    best_practices: list[str] = dspy.OutputField(desc="Best practices and tips")
    imports_needed: list[str] = dspy.OutputField(desc="Required imports and dependencies")

class DocumentationLearningAgent(dspy.Module):
    """Agent that learns from documentation URLs and generates code examples."""
    
    def __init__(self):
        super().__init__()
        self.fetcher = DocumentationFetcher()
        self.analyze_docs = dspy.ChainOfThought(LibraryAnalyzer)
        self.generate_code = dspy.ChainOfThought(CodeGenerator)
        self.refine_code = dspy.ChainOfThought(
            "code, feedback -> improved_code: str, changes_made: list[str]"
        )
    
    def learn_from_urls(self, library_name: str, doc_urls: list[str]) -> Dict:
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
def learn_library_from_urls(library_name: str, documentation_urls: list[str]) -> Dict:
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

## Step 4: Interactive Library Learning Function

```python
def learn_any_library(library_name: str, documentation_urls: list[str], use_cases: list[str] = None):
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

def interactive_learning_session():
    """Interactive session for learning libraries with user input."""
    
    print("🎯 Welcome to the Interactive Library Learning System!")
    print("This system will help you learn any Python library from its documentation.\n")
    
    learned_libraries = {}
    
    while True:
        print("\n" + "="*60)
        print("🚀 LIBRARY LEARNING SESSION")
        print("="*60)
        
        # Get library name from user
        library_name = input("\n📚 Enter the library name you want to learn (or 'quit' to exit): ").strip()
        
        if library_name.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Interactive Library Learning System!")
            break
        
        if not library_name:
            print("❌ Please enter a valid library name.")
            continue
        
        # Get documentation URLs
        print(f"\n🔗 Enter documentation URLs for {library_name} (one per line, empty line to finish):")
        urls = []
        while True:
            url = input("  URL: ").strip()
            if not url:
                break
            if not url.startswith(('http://', 'https://')):
                print("    ⚠️  Please enter a valid URL starting with http:// or https://")
                continue
            urls.append(url)
        
        if not urls:
            print("❌ No valid URLs provided. Skipping this library.")
            continue
        
        # Get custom use cases from user
        print(f"\n🎯 Define use cases for {library_name} (optional, press Enter for defaults):")
        print("   Default use cases will be: Basic setup, Common operations, Advanced usage")
        
        user_wants_custom = input("   Do you want to define custom use cases? (y/n): ").strip().lower()
        
        use_cases = None
        if user_wants_custom in ['y', 'yes']:
            print("   Enter your use cases (one per line, empty line to finish):")
            use_cases = []
            while True:
                use_case = input("     Use case: ").strip()
                if not use_case:
                    break
                use_cases.append(use_case)
            
            if not use_cases:
                print("   No custom use cases provided, using defaults.")
                use_cases = None
        
        # Learn the library
        print(f"\n🚀 Starting learning process for {library_name}...")
        result = learn_any_library(library_name, urls, use_cases)
        
        if result:
            learned_libraries[library_name] = result
            print(f"\n✅ Successfully learned {library_name}!")
            
            # Show summary
            print(f"\n📊 Learning Summary for {library_name}:")
            print(f"   • Core concepts: {len(result['library_info']['core_concepts'])} identified")
            print(f"   • Common patterns: {len(result['library_info']['patterns'])} found")
            print(f"   • Examples generated: {len(result['examples'])}")
            
            # Ask if user wants to see examples
            show_examples = input(f"\n👀 Do you want to see the generated examples for {library_name}? (y/n): ").strip().lower()
            
            if show_examples in ['y', 'yes']:
                for i, example in enumerate(result['examples'], 1):
                    print(f"\n{'─'*50}")
                    print(f"📝 Example {i}: {example['use_case']}")
                    print(f"{'─'*50}")
                    
                    print("\n💻 Generated Code:")
                    print("```python")
                    print(example['code'])
                    print("```")
                    
                    print(f"\n📦 Required Imports:")
                    for imp in example['imports']:
                        print(f"  • {imp}")
                    
                    print(f"\n📝 Explanation:")
                    print(example['explanation'])
                    
                    print(f"\n✅ Best Practices:")
                    for practice in example['best_practices']:
                        print(f"  • {practice}")
                    
                    # Ask if user wants to see the next example
                    if i < len(result['examples']):
                        continue_viewing = input(f"\nContinue to next example? (y/n): ").strip().lower()
                        if continue_viewing not in ['y', 'yes']:
                            break
            
            # Offer to save results
            save_results = input(f"\n💾 Save learning results for {library_name} to file? (y/n): ").strip().lower()
            
            if save_results in ['y', 'yes']:
                filename = input(f"   Enter filename (default: {library_name.lower()}_learning.json): ").strip()
                if not filename:
                    filename = f"{library_name.lower()}_learning.json"
                
                try:
                    import json
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"   ✅ Results saved to {filename}")
                except Exception as e:
                    print(f"   ❌ Error saving file: {e}")
        
        else:
            print(f"❌ Failed to learn {library_name}")
        
        # Ask if user wants to learn another library
        print(f"\n📚 Libraries learned so far: {list(learned_libraries.keys())}")
        continue_learning = input("\n🔄 Do you want to learn another library? (y/n): ").strip().lower()
        
        if continue_learning not in ['y', 'yes']:
            break
    
    # Final summary
    if learned_libraries:
        print(f"\n🎉 Session Summary:")
        print(f"Successfully learned {len(learned_libraries)} libraries:")
        for lib_name, info in learned_libraries.items():
            print(f"  • {lib_name}: {len(info['examples'])} examples generated")
    
    return learned_libraries

# Example: Run interactive learning session
if __name__ == "__main__":
    # Run interactive session
    learned_libraries = interactive_learning_session()
```

## Example Output

When you run the interactive learning system, you'll see:

**Interactive Session Start:**
```
🎯 Welcome to the Interactive Library Learning System!
This system will help you learn any Python library from its documentation.

============================================================
🚀 LIBRARY LEARNING SESSION
============================================================

📚 Enter the library name you want to learn (or 'quit' to exit): FastAPI

🔗 Enter documentation URLs for FastAPI (one per line, empty line to finish):
  URL: https://fastapi.tiangolo.com/
  URL: https://fastapi.tiangolo.com/tutorial/first-steps/
  URL: https://fastapi.tiangolo.com/tutorial/path-params/
  URL: 

🎯 Define use cases for FastAPI (optional, press Enter for defaults):
   Default use cases will be: Basic setup, Common operations, Advanced usage
   Do you want to define custom use cases? (y/n): y
   Enter your use cases (one per line, empty line to finish):
     Use case: Create a REST API with authentication
     Use case: Build a file upload endpoint
     Use case: Add database integration with SQLAlchemy
     Use case: 
```

**Documentation Processing:**
```
🚀 Starting learning process for FastAPI...
🚀 Starting automated learning for FastAPI...
Documentation sources: 3 URLs
📡 Fetching: https://fastapi.tiangolo.com/ (attempt 1)
📡 Fetching: https://fastapi.tiangolo.com/tutorial/first-steps/ (attempt 1)
📡 Fetching: https://fastapi.tiangolo.com/tutorial/path-params/ (attempt 1)
📚 Learning about FastAPI from 3 URLs...

🔍 Library Analysis Results for FastAPI:
Sources: 3 successful fetches
Core Concepts: ['FastAPI app', 'path operations', 'dependencies', 'request/response models']
Common Patterns: ['app = FastAPI()', 'decorator-based routing', 'Pydantic models']
Key Methods: ['FastAPI()', '@app.get()', '@app.post()', 'uvicorn.run()']
Installation: pip install fastapi uvicorn
```

**Code Generation:**
```
📝 Generating example 1/3: Create a REST API with authentication

✅ Successfully learned FastAPI!

📊 Learning Summary for FastAPI:
   • Core concepts: 4 identified
   • Common patterns: 3 found
   • Examples generated: 3

👀 Do you want to see the generated examples for FastAPI? (y/n): y

──────────────────────────────────────────────────
📝 Example 1: Create a REST API with authentication
──────────────────────────────────────────────────

💻 Generated Code:
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from typing import Dict
import jwt
from datetime import datetime, timedelta

app = FastAPI(title="Authenticated API", version="1.0.0")
security = HTTPBearer()

# Secret key for JWT (use environment variable in production)
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/login")
async def login(username: str, password: str) -> dict[str, str]:
    # In production, verify against database
    if username == "admin" and password == "secret":
        token_data = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=24)}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/protected")
async def protected_route(current_user: str = Depends(verify_token)) -> dict[str, str]:
    return {"message": f"Hello {current_user}! This is a protected route."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

📦 Required Imports:
  • pip install fastapi uvicorn python-jose[cryptography]
  • from fastapi import FastAPI, Depends, HTTPException, status
  • from fastapi.security import HTTPBearer
  • import jwt

📝 Explanation:
This example creates a FastAPI application with JWT-based authentication. It includes a login endpoint that returns a JWT token and a protected route that requires authentication...

✅ Best Practices:
  • Use environment variables for secret keys
  • Implement proper password hashing in production
  • Add token expiration and refresh logic
  • Include proper error handling

Continue to next example? (y/n): n

💾 Save learning results for FastAPI to file? (y/n): y
   Enter filename (default: fastapi_learning.json): 
   ✅ Results saved to fastapi_learning.json

📚 Libraries learned so far: ['FastAPI']

🔄 Do you want to learn another library? (y/n): n

🎉 Session Summary:
Successfully learned 1 libraries:
  • FastAPI: 3 examples generated
```


## Next Steps

- **GitHub Integration**: Learn from README files and example repositories
- **Video Tutorial Processing**: Extract information from video documentation
- **Community Examples**: Aggregate examples from Stack Overflow and forums
- **Version Comparison**: Track API changes across library versions
- **Testing Generation**: Automatically create unit tests for generated code
- **Page Crawling**: Automatically crawl documentation pages to actively understand the usage

This tutorial demonstrates how DSPy can automate the entire process of learning unfamiliar libraries from their documentation, making it valuable for rapid technology adoption and exploration.
