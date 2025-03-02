#!/usr/bin/env python3
"""
MinionsLM: Intelligent routing between local and remote language models.

This module implements the MinionsLM class, which provides intelligent
routing between local and remote language models based on query complexity.
It also handles document chunking, parallel processing, and error recovery.
"""

import time
import re
import concurrent.futures
from typing import Dict, List, Any, Optional

from dspy.clients.base_lm import BaseLM


class MinionsLM(BaseLM):
    """
    MinionsLM: Intelligently routes queries between local and remote language models.
    
    This class implements the Minions protocol, routing simpler queries to a local LM
    and more complex ones to a more capable remote LM. It also handles document
    chunking, parallel processing, and error recovery.
    
    Attributes:
        local_lm: The local language model for simpler queries
        remote_lm: The remote language model for complex queries
        complexity_threshold: Threshold for determining query complexity (0.0-1.0)
        max_parallel_chunks: Maximum number of chunks to process in parallel
        max_retries: Maximum number of retries on model errors
        verbose: Whether to print detailed logs
        metrics: Dictionary for tracking performance metrics
    """
    
    def __init__(
        self,
        local_lm: BaseLM,
        remote_lm: BaseLM,
        complexity_threshold: float = 0.6,
        max_parallel_chunks: int = 3,
        max_retries: int = 2,
        verbose: bool = False
    ) -> None:
        """
        Initialize the MinionsLM with local and remote language models.
        
        Args:
            local_lm: The local language model for simpler queries
            remote_lm: The remote language model for complex queries
            complexity_threshold: Threshold for determining query complexity (0.0-1.0)
            max_parallel_chunks: Maximum number of chunks to process in parallel
            max_retries: Maximum number of retries on model errors
            verbose: Whether to print detailed logs
        """
        # Store LMs and parameters
        self.local_lm = local_lm
        self.remote_lm = remote_lm
        self.complexity_threshold = complexity_threshold
        self.max_parallel_chunks = max_parallel_chunks
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Initialize metrics
        self.reset_metrics()
        
        # Call the parent constructor
        super().__init__(model=f"minions({local_lm.model},{remote_lm.model})")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics to their initial values."""
        self.metrics = {
            "local_calls": 0,
            "remote_calls": 0,
            "total_time": 0.0,
            "local_time": 0.0,
            "remote_time": 0.0,
            "retry_count": 0,
            "complexity_scores": [],
            "chunks_processed": 0,
        }
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Process a query using the appropriate language model.
        
        Args:
            prompt: Text prompt to process
            messages: List of chat messages to process
            **kwargs: Additional arguments to pass to the language model
            
        Returns:
            A list of response strings
        """
        start_time = time.time()
        
        try:
            # Handle chat messages differently (always use local model for simplicity)
            if messages is not None:
                if self.verbose:
                    print("MinionsLM: Processing chat messages using local model")
                
                self.metrics["local_calls"] += 1
                local_start = time.time()
                response = self._call_with_retry(self.local_lm, messages=messages, **kwargs)
                self.metrics["local_time"] += time.time() - local_start
                
                self.metrics["total_time"] = time.time() - start_time
                return response
            
            # Handle text prompts
            if prompt is None:
                prompt = ""
            
            # Check if we need to chunk the document
            estimated_tokens = self._estimate_tokens(prompt)
            if estimated_tokens > 1500:  # Threshold for chunking
                if self.verbose:
                    print(f"MinionsLM: Document exceeds token limit ({estimated_tokens} tokens). Chunking...")
                
                return self._process_long_document(prompt, **kwargs)
            
            # Assess complexity for routing decision
            complexity = self._assess_complexity(prompt)
            self.metrics["complexity_scores"].append(complexity)
            
            # Route to appropriate model based on complexity
            if complexity < self.complexity_threshold:
                if self.verbose:
                    print(f"MinionsLM: Routing to local model (complexity: {complexity:.2f})")
                
                self.metrics["local_calls"] += 1
                local_start = time.time()
                response = self._call_with_retry(self.local_lm, prompt=prompt, **kwargs)
                self.metrics["local_time"] += time.time() - local_start
            else:
                if self.verbose:
                    print(f"MinionsLM: Routing to remote model (complexity: {complexity:.2f})")
                
                self.metrics["remote_calls"] += 1
                remote_start = time.time()
                response = self._call_with_retry(self.remote_lm, prompt=prompt, **kwargs)
                self.metrics["remote_time"] += time.time() - remote_start
            
            # Update total time
            self.metrics["total_time"] = time.time() - start_time
            
            return response
        
        except Exception as e:
            # Update total time even if we encounter an error
            self.metrics["total_time"] = time.time() - start_time
            
            # Re-raise the exception
            raise e
    
    def _call_with_retry(
        self,
        model: BaseLM,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Call a language model with retry logic.
        
        Args:
            model: The language model to call
            prompt: Optional prompt text
            messages: Optional chat messages
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            A list of responses
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if prompt is not None:
                    response = model(prompt=prompt, **kwargs)
                else:
                    response = model(messages=messages, **kwargs)
                
                # Check for empty or None responses
                if not response or not any(r for r in response if r):
                    raise ValueError("Empty response received from model")
                
                return response
            
            except Exception as e:
                last_error = e
                self.metrics["retry_count"] += 1
                
                if self.verbose:
                    print(f"MinionsLM: Error on attempt {attempt+1}/{self.max_retries+1}: {str(e)}")
                
                if attempt < self.max_retries:
                    # Exponential backoff (wait 1s, then 2s, then 4s, etc.)
                    backoff_time = 2 ** attempt
                    time.sleep(backoff_time)
                    
                    if self.verbose:
                        print(f"MinionsLM: Retrying after {backoff_time}s...")
        
        # If we've exhausted all retries, return an error message or fallback to the other model
        if self.verbose:
            print("MinionsLM: All retries failed. Using fallback error message.")
        
        error_message = f"Error: Failed to get response after {self.max_retries} retries. Last error: {str(last_error)}"
        return [error_message]
    
    def _process_long_document(self, document: str, **kwargs: Any) -> List[str]:
        """
        Process long documents by chunking and parallel processing.
        
        Args:
            document: The long document to process
            **kwargs: Additional arguments to pass to the language model
            
        Returns:
            A list containing the aggregated response
        """
        if self.verbose:
            print("MinionsLM: Processing long document...")
        
        # Chunk the document
        chunks = self._chunk_document(document)
        self.metrics["chunks_processed"] += len(chunks)
        
        if self.verbose:
            print(f"MinionsLM: Document split into {len(chunks)} chunks")
        
        # Process chunks in parallel
        chunk_responses = []
        max_workers = min(self.max_parallel_chunks, len(chunks))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a list of futures for each chunk
            futures = {
                executor.submit(self._process_chunk, chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    chunk_responses.append((chunk_idx, result))
                    
                    if self.verbose:
                        print(f"MinionsLM: Processed chunk {chunk_idx + 1}/{len(chunks)}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"MinionsLM: Error processing chunk {chunk_idx + 1}: {str(e)}")
                    
                    # Add an error placeholder for this chunk
                    chunk_responses.append((chunk_idx, [f"[Error processing chunk {chunk_idx + 1}: {str(e)}]"]))
        
        # Sort responses by their original chunk index
        chunk_responses.sort(key=lambda x: x[0])
        sorted_responses = [resp for _, resp in chunk_responses]
        
        # Convert list of response lists to a single list of strings (flatten)
        flattened_responses = [item for sublist in sorted_responses for item in sublist]
        
        # For very long documents, synthesize a final summary using the remote model
        if len(chunks) > 3:
            if self.verbose:
                print("MinionsLM: Synthesizing final summary...")
            
            # Create a prompt that asks for synthesis of the responses
            synthesis_prompt = (
                "Below are summaries of different sections of a long document. "
                "Please synthesize these into a cohesive, integrated response that "
                "captures the key information:\n\n" + 
                "\n\n".join([f"Section {i+1}: {resp[0]}" for i, resp in enumerate(sorted_responses)])
            )
            
            self.metrics["remote_calls"] += 1
            remote_start = time.time()
            final_response = self._call_with_retry(self.remote_lm, prompt=synthesis_prompt, **kwargs)
            self.metrics["remote_time"] += time.time() - remote_start
            
            return final_response
        
        # For shorter documents, just concatenate the responses
        return [" ".join(flattened_responses)]
    
    def _process_chunk(self, chunk: str, **kwargs: Any) -> List[str]:
        """
        Process a single document chunk.
        
        Args:
            chunk: The text chunk to process
            **kwargs: Additional arguments to pass to the language model
            
        Returns:
            The language model's response for this chunk
        """
        # Assess complexity
        complexity = self._assess_complexity(chunk)
        self.metrics["complexity_scores"].append(complexity)
        
        # Route based on complexity
        if complexity < self.complexity_threshold:
            if self.verbose:
                print(f"MinionsLM: Processing chunk with local model (complexity: {complexity:.2f})")
            
            self.metrics["local_calls"] += 1
            local_start = time.time()
            response = self._call_with_retry(self.local_lm, prompt=chunk, **kwargs)
            self.metrics["local_time"] += time.time() - local_start
        else:
            if self.verbose:
                print(f"MinionsLM: Processing chunk with remote model (complexity: {complexity:.2f})")
            
            self.metrics["remote_calls"] += 1
            remote_start = time.time()
            response = self._call_with_retry(self.remote_lm, prompt=chunk, **kwargs)
            self.metrics["remote_time"] += time.time() - remote_start
        
        return response
    
    def _assess_complexity(self, text: str) -> float:
        """
        Assess the complexity of a text to determine routing.
        
        Args:
            text: The text to assess
            
        Returns:
            A complexity score between 0.0 and 1.0
        """
        # Implementation can be extended for more sophisticated complexity assessment
        # Current implementation uses simple heuristics
        
        # Normalize to lower case for analysis
        normalized_text = text.lower()
        
        # 1. Length-based complexity (longer texts are generally more complex)
        length_score = min(len(text) / 5000, 1.0) * 0.3
        
        # 2. Vocabulary-based complexity
        complex_terms = [
            'quantum', 'algorithm', 'neural', 'network', 'blockchain', 'cryptocurrency',
            'philosophy', 'mathematical', 'theorem', 'hypothesis', 'synthesis',
            'implementation', 'architecture', 'infrastructure', 'paradigm', 'methodology',
            'theoretical', 'simulation', 'empirical', 'statistical', 'analytical',
            'optimization', 'integration', 'differential', 'cognitive', 'linguistic'
        ]
        
        term_count = sum(1 for term in complex_terms if term in normalized_text)
        term_score = min(term_count / 10, 1.0) * 0.3
        
        # 3. Structure-based complexity
        structure_indicators = [
            '```', 'def ', 'class ', 'function', 'algorithm', 'code',
            'pseudocode', 'implementation', 'for ', 'while ', 'if ', 'else',
            'return ', 'import ', '@', 'package', 'module'
        ]
        
        structure_count = sum(1 for indicator in structure_indicators if indicator in normalized_text)
        structure_score = min(structure_count / 8, 1.0) * 0.2
        
        # 4. Question complexity
        question_indicators = [
            'explain', 'analyze', 'compare', 'contrast', 'evaluate', 'synthesize',
            'justify', 'critique', 'assess', 'examine', 'investigate', 'interpret',
            'elaborate', 'implications', 'significance', 'meaning'
        ]
        
        question_count = sum(1 for indicator in question_indicators if indicator in normalized_text)
        question_score = min(question_count / 5, 1.0) * 0.2
        
        # Final score is a weighted sum of the individual scores
        final_score = length_score + term_score + structure_score + question_score
        
        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # A very simple token estimation heuristic
        # Actual tokenization depends on the specific tokenizer used by the model
        if not text:
            return 0
        
        # Approximately 4 characters per token for English text (rough estimate)
        return len(text) // 4
    
    def _chunk_document(self, document: str) -> List[str]:
        """
        Split a document into manageable chunks, preserving structure.
        
        Args:
            document: The document to chunk
            
        Returns:
            A list of document chunks
        """
        # First, try to split by natural document sections (headings)
        sections = re.split(r'(?m)^#{1,3} ', document)
        
        # If we have sections and they're not too big, use them
        if len(sections) > 1 and all(len(s) < 4000 for s in sections):
            # Reattach the headings that were removed in the split
            chunks = []
            for i, section in enumerate(sections):
                if i == 0 and not section.strip():
                    continue  # Skip empty first section
                
                # For all non-first sections, prepend the heading marker
                if i > 0:
                    chunks.append("# " + section)
                else:
                    chunks.append(section)
            
            return chunks
        
        # Otherwise, split by paragraphs
        paragraphs = re.split(r'\n\s*\n', document)
        
        # Group paragraphs into chunks of approximately 1000 tokens
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_tokens = 1000
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # If this paragraph alone exceeds the chunk size, split it further
            if para_tokens > max_chunk_tokens:
                # If we have content in the current chunk, add it first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    sentence_tokens = self._estimate_tokens(sentence)
                    
                    if sentence_length + sentence_tokens > max_chunk_tokens:
                        if sentence_chunk:
                            chunks.append(' '.join(sentence_chunk))
                            sentence_chunk = []
                            sentence_length = 0
                    
                    sentence_chunk.append(sentence)
                    sentence_length += sentence_tokens
                
                # Add any remaining sentences
                if sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))
            
            # Normal case: If adding this paragraph exceeds the chunk size, start a new chunk
            elif current_length + para_tokens > max_chunk_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_length = para_tokens
                else:
                    # This shouldn't happen normally, but just in case
                    chunks.append(para)
                    current_chunk = []
                    current_length = 0
            
            # Add to current chunk
            else:
                current_chunk.append(para)
                current_length += para_tokens
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


def create_minions_lm(
    local_lm: BaseLM,
    remote_lm: BaseLM,
    complexity_threshold: float = 0.6,
    max_parallel_chunks: int = 3,
    max_retries: int = 2,
    verbose: bool = False
) -> MinionsLM:
    """
    Create a MinionsLM instance with the specified configuration.
    
    Args:
        local_lm: The local language model for simpler queries
        remote_lm: The remote language model for complex queries
        complexity_threshold: Threshold for determining query complexity (0.0-1.0)
        max_parallel_chunks: Maximum number of chunks to process in parallel
        max_retries: Maximum number of retries on model errors
        verbose: Whether to print detailed logs
        
    Returns:
        A configured MinionsLM instance
    """
    return MinionsLM(
        local_lm=local_lm,
        remote_lm=remote_lm,
        complexity_threshold=complexity_threshold,
        max_parallel_chunks=max_parallel_chunks,
        max_retries=max_retries,
        verbose=verbose
    )

# Example signatures are moved to the demo script to avoid circular imports 