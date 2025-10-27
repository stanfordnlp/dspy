"""Benchmark for cache key generation performance (SHA256 vs xxhash)."""
import time
from hashlib import sha256

import orjson
import pytest
import xxhash


def benchmark_hash_function(hash_func, data_samples, iterations=1000):
    """Benchmark a hash function with multiple data samples."""
    # Warmup
    for _ in range(10):
        for data in data_samples:
            hash_func(data)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        for data in data_samples:
            hash_func(data)
    duration = time.perf_counter() - start
    
    return duration / (iterations * len(data_samples))


def create_test_data():
    """Create realistic cache request data samples."""
    return [
        # Small request
        orjson.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}, option=orjson.OPT_SORT_KEYS),
        
        # Medium request
        orjson.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?" * 10}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }, option=orjson.OPT_SORT_KEYS),
        
        # Large request
        orjson.dumps({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in detail." * 50}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5
        }, option=orjson.OPT_SORT_KEYS),
        
        # Request with nested structures
        orjson.dumps({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "test" * 20}
            ],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "description": "Get weather data"}},
                {"type": "function", "function": {"name": "search", "description": "Search the web"}}
            ]
        }, option=orjson.OPT_SORT_KEYS),
    ]


def test_sha256_performance():
    """Benchmark SHA256 hash performance (current implementation)."""
    data_samples = create_test_data()
    
    def sha256_hash(data):
        return sha256(data).hexdigest()
    
    avg_time = benchmark_hash_function(sha256_hash, data_samples, iterations=1000)
    
    print(f"\nSHA256 average time: {avg_time*1e6:.2f}µs per operation")
    assert avg_time > 0  # Sanity check


def test_xxhash_performance():
    """Benchmark xxhash performance (proposed implementation)."""
    data_samples = create_test_data()
    
    def xxhash_hash(data):
        return xxhash.xxh64(data).hexdigest()
    
    avg_time = benchmark_hash_function(xxhash_hash, data_samples, iterations=1000)
    
    print(f"\nxxhash average time: {avg_time*1e6:.2f}µs per operation")
    assert avg_time > 0  # Sanity check


def test_hash_performance_comparison():
    """Compare SHA256 vs xxhash performance."""
    data_samples = create_test_data()
    
    def sha256_hash(data):
        return sha256(data).hexdigest()
    
    def xxhash_hash(data):
        return xxhash.xxh64(data).hexdigest()
    
    sha256_time = benchmark_hash_function(sha256_hash, data_samples, iterations=1000)
    xxhash_time = benchmark_hash_function(xxhash_hash, data_samples, iterations=1000)
    
    speedup = sha256_time / xxhash_time
    
    print(f"\n{'='*70}")
    print("CACHE KEY GENERATION PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"SHA256 (current):  {sha256_time*1e6:.2f}µs per operation")
    print(f"xxhash (proposed): {xxhash_time*1e6:.2f}µs per operation")
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Time reduction: {((sha256_time - xxhash_time)/sha256_time)*100:.1f}%")
    print(f"{'='*70}")
    
    # xxhash should be significantly faster
    assert xxhash_time < sha256_time, "xxhash should be faster than SHA256"
    assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.2f}x"


def test_real_cache_usage_pattern():
    """Benchmark realistic cache usage pattern with multiple requests."""
    import dspy
    
    # Simulate cache key generation for typical LM requests
    requests = [
        {"model": "gpt-4", "messages": [{"role": "user", "content": f"Question {i}"}], "temperature": 0.7}
        for i in range(100)
    ]
    
    cache = dspy.cache
    
    # Benchmark SHA256 (current)
    start = time.perf_counter()
    for request in requests:
        _ = cache.cache_key(request)
    sha256_total = time.perf_counter() - start
    
    print(f"\n100 cache key generations:")
    print(f"  SHA256 total time: {sha256_total*1000:.2f}ms")
    print(f"  Average per key: {sha256_total*1000/100:.2f}ms")
    
    # This test just shows current performance baseline
    # After implementing xxhash, we'll add comparison here

