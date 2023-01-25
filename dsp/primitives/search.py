import dsp


def retrieve(query: str, k):
    passages = dsp.settings.rm(query, k=k)
    passages = [psg.long_text for psg in passages]

    return passages

def retrieveEnsemble(queries, k, by_prob=True):
    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k*3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    passages = sorted(passages, reverse=True)[:k]
    passages = [text for _, text in passages]

    return passages
