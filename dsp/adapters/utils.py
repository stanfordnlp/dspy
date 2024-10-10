from typing import Optional, Union


def passages2text(passages: Union[str, list, tuple]) -> str:
    """Formats the given one or more passages into a single structured string."""
    if isinstance(passages, str):
        return passages

    assert type(passages) in [list, tuple]

    if len(passages) == 0:
        return "N/A"

    if len(passages) == 1:
        return f"«{passages[0]}»"

    return "\n".join([f"[{idx+1}] «{txt}»" for idx, txt in enumerate(passages)])


# def passages2textV2(passages: Union[str, list, tuple]) -> str:
#     """Formats the given one or more passages into a single structured string."""
#     if isinstance(passages, str):
#         return passages

#     assert type(passages) in [list, tuple]

#     def psg2text(psg):
#         try:
#             title, snippet = psg.split("|", 1)
#             return f"Title: {title.strip()} | Snippet: «{snippet.strip()}»"
#         except Exception:
#             pass
        
#         return f"«{psg}»"

#     if len(passages) == 0:
#         return "N/A"

#     if len(passages) == 1:
#         return psg2text(passages[0])

#     return "\n".join([f"[{idx+1}] {psg2text(txt)}" for idx, txt in enumerate(passages)])


def format_answers(answers: Union[str, list]) -> Optional[str]:
    """Parses the given answers and returns the appropriate answer string.

    Args:
        answers (Union[str, list]): The answers to parse.

    Raises:
        ValueError: when instance is of type list and has no answers
        ValueError: when is not of type list or str

    Returns:
        _type_: Optiona[str]
    """
    if isinstance(answers, list):
        if len(answers) >= 1:
            return str(answers[0]).strip()
        if len(answers) == 0:
            raise ValueError("No answers found")
    elif isinstance(answers, str):
        return answers
    else:
        raise ValueError(f"Unable to parse answers of type {type(answers)}")
