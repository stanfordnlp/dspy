# Generating llms.txt for Code Documentation with DSPy

This tutorial demonstrates how to use DSPy to automatically generate an `llms.txt` file for the DSPy repository itself. The `llms.txt` standard provides LLM-friendly documentation that helps AI systems better understand codebases.

## What is llms.txt?

`llms.txt` is a proposed standard for providing structured, LLM-friendly documentation about a project. It typically includes:

- Project overview and purpose
- Key concepts and terminology
- Architecture and structure
- Usage examples
- Important files and directories

## Building a DSPy Program for llms.txt Generation

Let's create a DSPy program that analyzes a repository and generates comprehensive `llms.txt` documentation.

### Step 1: Define Our Signatures

First, we'll define signatures for different aspects of documentation generation:

```python
import dspy
from typing import List

class AnalyzeRepository(dspy.Signature):
    """Analyze a repository structure and identify key components."""
    repo_url: str = dspy.InputField(desc="GitHub repository URL")
    file_tree: str = dspy.InputField(desc="Repository file structure")
    readme_content: str = dspy.InputField(desc="README.md content")
    
    project_purpose: str = dspy.OutputField(desc="Main purpose and goals of the project")
    key_concepts: list[str] = dspy.OutputField(desc="List of important concepts and terminology")
    architecture_overview: str = dspy.OutputField(desc="High-level architecture description")

class AnalyzeCodeStructure(dspy.Signature):
    """Analyze code structure to identify important directories and files."""
    file_tree: str = dspy.InputField(desc="Repository file structure")
    package_files: str = dspy.InputField(desc="Key package and configuration files")
    
    important_directories: list[str] = dspy.OutputField(desc="Key directories and their purposes")
    entry_points: list[str] = dspy.OutputField(desc="Main entry points and important files")
    development_info: str = dspy.OutputField(desc="Development setup and workflow information")

class GenerateLLMsTxt(dspy.Signature):
    """Generate a comprehensive llms.txt file from analyzed repository information."""
    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="Common usage patterns and examples")
    
    llms_txt_content: str = dspy.OutputField(desc="Complete llms.txt file content following the standard format")
```

### Step 2: Create the Repository Analyzer Module

```python
class RepositoryAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepository)
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)
        self.generate_examples = dspy.ChainOfThought("repo_info -> usage_examples")
        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)
    
    def forward(self, repo_url, file_tree, readme_content, package_files):
        # Analyze repository purpose and concepts
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content
        )
        
        # Analyze code structure
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files
        )
        
        # Generate usage examples
        usage_examples = self.generate_examples(
            repo_info=f"Purpose: {repo_analysis.project_purpose}\nConcepts: {repo_analysis.key_concepts}"
        )
        
        # Generate final llms.txt
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples
        )
        
        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis
        )
```

### Step 3: Gather Repository Information

Let's create helper functions to extract repository information:

```python
import requests
import os
from pathlib import Path

os.environ["GITHUB_ACCESS_TOKEN"] = "<your_access_token>"

def get_github_file_tree(repo_url):
    """Get repository file structure from GitHub API."""
    # Extract owner/repo from URL
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"Failed to fetch repository tree: {response.status_code}")

def get_github_file_content(repo_url, file_path):
    """Get specific file content from GitHub."""
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        import base64
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content
    else:
        return f"Could not fetch {file_path}"

def gather_repository_info(repo_url):
    """Gather all necessary repository information."""
    file_tree = get_github_file_tree(repo_url)
    readme_content = get_github_file_content(repo_url, "README.md")
    
    # Get key package files
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json"]:
        try:
            content = get_github_file_content(repo_url, file_path)
            if "Could not fetch" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except:
            continue
    
    package_files_content = "\n\n".join(package_files)
    
    return file_tree, readme_content, package_files_content
```

### Step 4: Configure DSPy and Generate llms.txt

```python
def generate_llms_txt_for_dspy():
    # Configure DSPy (use your preferred LM)
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"
    
    # Initialize our analyzer
    analyzer = RepositoryAnalyzer()
    
    # Gather DSPy repository information
    repo_url = "https://github.com/stanfordnlp/dspy"
    file_tree, readme_content, package_files = gather_repository_info(repo_url)
    
    # Generate llms.txt
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files
    )
    
    return result

# Run the generation
if __name__ == "__main__":
    result = generate_llms_txt_for_dspy()
    
    # Save the generated llms.txt
    with open("llms.txt", "w") as f:
        f.write(result.llms_txt_content)
    
    print("Generated llms.txt file!")
    print("\nPreview:")
    print(result.llms_txt_content[:500] + "...")
```

## Expected Output Structure

The generated `llms.txt` for DSPy would follow this structure:

```
# DSPy: Programming Language Models

## Project Overview
DSPy is a framework for programming—rather than prompting—language models...

## Key Concepts
- **Modules**: Building blocks for LM programs
- **Signatures**: Input/output specifications  
- **Teleprompters**: Optimization algorithms
- **Predictors**: Core reasoning components

## Architecture
- `/dspy/`: Main package directory
  - `/adapters/`: Input/output format handlers
  - `/clients/`: LM client interfaces
  - `/predict/`: Core prediction modules
  - `/teleprompt/`: Optimization algorithms

## Usage Examples
1. **Building a Classifier**: Using DSPy, a user can define a modular classifier that takes in text data and categorizes it into predefined classes. The user can specify the classification logic declaratively, allowing for easy adjustments and optimizations.
2. **Creating a RAG Pipeline**: A developer can implement a retrieval-augmented generation pipeline that first retrieves relevant documents based on a query and then generates a coherent response using those documents. DSPy facilitates the integration of retrieval and generation components seamlessly.
3. **Optimizing Prompts**: Users can leverage DSPy to create a system that automatically optimizes prompts for language models based on performance metrics, improving the quality of responses over time without manual intervention.
4. **Implementing Agent Loops**: A user can design an agent loop that continuously interacts with users, learns from feedback, and refines its responses, showcasing the self-improving capabilities of the DSPy framework.
5. **Compositional Code**: Developers can write compositional code that allows different modules of the AI system to interact with each other, enabling complex workflows that can be easily modified and extended.
```

The resulting `llms.txt` file provides a comprehensive, LLM-friendly overview of the DSPy repository that can help other AI systems better understand and work with the codebase.

## Next Steps

- Extend the program to analyze multiple repositories
- Add support for different documentation formats
- Create metrics for documentation quality assessment
- Build a web interface for interactive repository analysis
