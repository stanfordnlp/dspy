def count_messages_with_file_pattern(messages):
    """Count content parts matching {"type": "file", "file": <dict>} in messages."""
    pattern = {"type": "file", "file": lambda x: isinstance(x, dict)}

    def check_pattern(obj, pattern):
        if isinstance(pattern, dict):
            if not isinstance(obj, dict):
                return False
            return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
        if callable(pattern):
            return pattern(obj)
        return obj == pattern

    def count_patterns(obj, pattern):
        count = 0
        if check_pattern(obj, pattern):
            count += 1
        if isinstance(obj, dict):
            count += sum(count_patterns(v, pattern) for v in obj.values())
        if isinstance(obj, list | tuple):
            count += sum(count_patterns(v, pattern) for v in obj)
        return count

    return count_patterns(messages, pattern)
