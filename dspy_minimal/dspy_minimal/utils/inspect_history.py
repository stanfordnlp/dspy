def pretty_print_history(history, n=1):
    """Pretty print the history of LM calls."""
    if not history:
        return "No history available."
    
    result = []
    for i, entry in enumerate(history[-n:]):
        result.append(f"Entry {i+1}: {entry}")
    
    return "\n".join(result) 