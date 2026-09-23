"""Stand-in System One clients for the ReAnchor tests."""

import copy


class FakeClient:
    """Answers each question with `answer(state, name, question)`, and records every request."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, state, questions):
        self.calls.append(copy.deepcopy((state, questions)))
        return {name: self.answer(state, name, q) for name, q in questions.items()}

    async def acall(self, state, questions):
        return self(state, questions)


def noul(p: float) -> dict:
    return {"noul": p}


def score(probabilities: dict[int, float], confidence: float = 0.8) -> dict:
    return {"score": 0.0, "confidence": confidence, "probabilities": probabilities}


def choice(probabilities: dict[str, float], confidence: float = 0.7) -> dict:
    return {
        "choice": max(probabilities, key=probabilities.get),
        "confidence": confidence,
        "probabilities": probabilities,
    }

