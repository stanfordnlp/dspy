def passages2text(passages):
    if type(passages) is str:
        return passages

    assert type(passages) in [list, tuple]

    if len(passages) == 1:
        return f"«{passages[0]}»"

    return '\n'.join([f"[{idx+1}] «{txt}»" for idx, txt in enumerate(passages)])
