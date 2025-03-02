#!/usr/bin/env python3
"""
StructuredMinionsLM: Enhanced structured output capabilities for MinionsLM.

This module provides the StructuredMinionsLM class, which extends the base
MinionsLM to provide structured data extraction capabilities, handling queries,
document analysis, and specialized data extraction.
"""

import json
import re
from typing import Dict, List, Any, Optional

from dspy.clients.base_lm import BaseLM
from dspy.clients import MinionsLM, create_minions_lm


class StructuredMinionsLM:
    """
    StructuredMinionsLM provides structured data extraction using MinionsLM.
    
    This class extends the capabilities of MinionsLM to provide structured
    responses as dictionaries, with specialized methods for document analysis,
    recipe extraction, and other structured data extraction tasks.
    
    Attributes:
        minions_lm: The underlying MinionsLM instance
        verbose: Whether to print detailed logs
    """
    
    def __init__(
        self,
        minions_lm: MinionsLM,
        verbose: bool = False
    ) -> None:
        """
        Initialize the StructuredMinionsLM with a MinionsLM instance.
        
        Args:
            minions_lm: An initialized MinionsLM instance
            verbose: Whether to print detailed logs
        """
        self.minions_lm = minions_lm
        self.verbose = verbose
    
    def ask(self, question: str) -> str:
        """
        Process a simple question and return a text response.
        
        Args:
            question: The question to answer
            
        Returns:
            The text response
        """
        if self.verbose:
            print(f"StructuredMinionsLM: Processing question: {question}")
        
        response = self.minions_lm(prompt=question)
        
        if not response or not response[0].strip():
            return "Error: Unable to generate a response."
        
        return response[0]
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return a structured response as a dictionary.
        
        Args:
            question: The question to answer
            
        Returns:
            A dictionary containing the structured response
        """
        if self.verbose:
            print(f"StructuredMinionsLM: Processing structured query: {question}")
        
        prompt = f"""
        Answer the following question and provide your response in JSON format:
        
        Question: {question}
        
        Format your response as a JSON object with the following fields:
        {{
            "answer": "Your detailed answer to the question",
            "confidence": "A number between 0 and 1 indicating confidence in your answer",
            "relevant_facts": ["List of facts relevant to the question"]
        }}
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """
        Analyze a document and return a structured analysis.
        
        Args:
            document: The document text to analyze
            
        Returns:
            A dictionary containing document analysis results
        """
        if self.verbose:
            print(f"StructuredMinionsLM: Processing document of length {len(document)} characters")
        
        prompt = f"""
        Analyze the following document and provide your analysis in JSON format:
        
        {document}
        
        Return your analysis as a JSON object with the following fields:
        {{
            "summary": "A concise summary of the document",
            "key_points": ["List of key points from the document"],
            "topics": ["List of main topics covered"],
            "sentiment": "Overall sentiment of the document (positive, negative, neutral)",
            "entities": ["List of important entities mentioned"]
        }}
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def process_recipe(self, recipe_text: str) -> Dict[str, Any]:
        """
        Extract structured information from a recipe text.
        
        Args:
            recipe_text: The recipe text to process
            
        Returns:
            A dictionary containing structured recipe information
        """
        if self.verbose:
            print("StructuredMinionsLM: Extracting recipe information")
        
        prompt = f"""
        Extract structured information from the following recipe and provide the results in JSON format:
        
        {recipe_text}
        
        Return your extraction as a JSON object with the following fields:
        {{
            "title": "Recipe title",
            "ingredients": ["List of ingredients with quantities"],
            "instructions": ["Numbered list of preparation steps"],
            "prep_time": "Preparation time",
            "cook_time": "Cooking time",
            "yield": "Number of servings",
            "cuisine": "Type of cuisine (if determinable)"
        }}
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code and return structured information about it.
        
        Args:
            code: The code to analyze
            
        Returns:
            A dictionary containing code analysis results
        """
        if self.verbose:
            print(f"StructuredMinionsLM: Analyzing code of length {len(code)} characters")
        
        prompt = f"""
        Analyze the following code and provide your analysis in JSON format:
        
        ```
        {code}
        ```
        
        Return your analysis as a JSON object with the following fields:
        {{
            "language": "Programming language of the code",
            "summary": "A concise summary of what the code does",
            "functions": ["List of functions/methods defined"],
            "classes": ["List of classes defined"],
            "dependencies": ["List of imports/dependencies"],
            "potential_issues": ["List of potential issues or bugs"],
            "complexity_assessment": "Assessment of code complexity (simple, moderate, complex)"
        }}
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def extract_data(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from text according to a specified schema.
        
        Args:
            text: The text to extract data from
            schema: Dictionary describing the data fields to extract
            
        Returns:
            A dictionary containing the extracted data
        """
        if self.verbose:
            print("StructuredMinionsLM: Extracting data with custom schema")
        
        # Convert schema to a JSON-like string representation
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""
        Extract structured information from the following text according to the provided schema:
        
        TEXT:
        {text}
        
        SCHEMA:
        {schema_str}
        
        Return your extraction as a JSON object matching the structure of the schema.
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def compare_entities(self, entity1: str, entity2: str, criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two entities across various criteria.
        
        Args:
            entity1: The first entity to compare
            entity2: The second entity to compare
            criteria: Optional list of specific criteria to compare on
            
        Returns:
            A dictionary containing comparison results
        """
        if self.verbose:
            print(f"StructuredMinionsLM: Comparing {entity1} and {entity2}")
        
        criteria_text = ""
        if criteria and len(criteria) > 0:
            criteria_text = "Focus on the following criteria: " + ", ".join(criteria)
        
        prompt = f"""
        Compare {entity1} and {entity2} and provide your comparison in JSON format.
        {criteria_text}
        
        Return your comparison as a JSON object with the following fields:
        {{
            "similarities": ["List of similarities between the entities"],
            "differences": ["List of differences between the entities"],
            "pros_entity1": ["List of advantages of {entity1}"],
            "pros_entity2": ["List of advantages of {entity2}"],
            "cons_entity1": ["List of disadvantages of {entity1}"],
            "cons_entity2": ["List of disadvantages of {entity2}"],
            "recommendation": "Overall recommendation or assessment"
        }}
        """
        
        response = self.minions_lm(prompt=prompt)
        return self._parse_json_response(response[0])
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a JSON response from the language model.
        
        Args:
            response: The raw response from the language model
            
        Returns:
            A dictionary parsed from the JSON response
        """
        # Clean up the response to extract JSON
        try:
            # Try to find JSON content within markdown code blocks
            json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', response)
            if json_match:
                json_content = json_match.group(1)
            else:
                # If no code blocks, try to find anything that looks like JSON
                json_content = response
                
                # Remove any text before the first `{` and after the last `}`
                first_brace = json_content.find('{')
                last_brace = json_content.rfind('}')
                
                if first_brace != -1 and last_brace != -1:
                    json_content = json_content[first_brace:last_brace+1]
            
            # Parse the JSON
            result = json.loads(json_content)
            return result
        
        except json.JSONDecodeError:
            if self.verbose:
                print(f"Failed to parse JSON response: {response}")
            
            # Return empty dict on error
            return {}
        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing response: {e}")
            
            # Return empty dict on error
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the underlying MinionsLM.
        
        Returns:
            A dictionary of performance metrics
        """
        return self.minions_lm.metrics


def create_structured_minions(
    local_lm: BaseLM,
    remote_lm: BaseLM,
    complexity_threshold: float = 0.6,
    max_parallel_chunks: int = 3,
    max_retries: int = 2,
    verbose: bool = False
) -> StructuredMinionsLM:
    """
    Create a StructuredMinionsLM instance with the specified configuration.
    
    Args:
        local_lm: The local language model for simpler queries
        remote_lm: The remote language model for complex queries
        complexity_threshold: Threshold for determining query complexity (0.0-1.0)
        max_parallel_chunks: Maximum number of chunks to process in parallel
        max_retries: Maximum number of retries on model errors
        verbose: Whether to print detailed logs
        
    Returns:
        A configured StructuredMinionsLM instance
    """
    # Create the underlying MinionsLM
    minions_lm = create_minions_lm(
        local_lm=local_lm,
        remote_lm=remote_lm,
        complexity_threshold=complexity_threshold,
        max_parallel_chunks=max_parallel_chunks,
        max_retries=max_retries,
        verbose=verbose
    )
    
    # Create and return the StructuredMinionsLM
    return StructuredMinionsLM(minions_lm=minions_lm, verbose=verbose) 