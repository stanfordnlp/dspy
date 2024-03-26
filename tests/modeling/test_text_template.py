from dspy import Example, Signature, InputField, OutputField
from dspy.modeling import TextTemplate


class Emotion(Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence = InputField()
    sentiment = OutputField()


class CheckCitationFaithfulness(Signature):
    """Verify that the text is based on the provided context."""

    context = InputField(desc="facts here are assumed to be true")
    text = InputField()
    faithfulness = OutputField(
        desc="True/False indicating if text is faithful to context"
    )


class COTCheckCitationFaithfulness(Signature):
    """Verify that the text is based on the provided context."""

    context = InputField(desc="facts here are assumed to be true")
    text = InputField()
    rationale = OutputField(
        desc="Think step by step in order to generate the faithfulness.",
    )
    faithfulness = OutputField(
        desc="True/False indicating if text is faithful to context"
    )


TEMPLATE_SCENARIOS = [
    {
        "signature": Emotion,
        "output": "Joy",
        "input_kwargs": {
            "sentence": "This is a positive test sentence.",
        },
        "output_kwargs": {"sentiment": "Joy"},
        "prompt": "Classify emotion among sadness, joy, love, anger, fear, surprise.\n\n---\n\nFollow the following format.\n\nSentence: ${sentence}\n\nSentiment: ${sentiment}\n\n---\n\nSentence: This is a positive test sentence.\n\nSentiment:",
    },
    {
        "signature": CheckCitationFaithfulness,
        "output": "False",
        "input_kwargs": {
            "context": [
                "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
            ],
            "text": "Lee scored 3 goals for Colchester United.",
        },
        "output_kwargs": {
            "faithfulness": "False",
        },
        "prompt": "Verify that the text is based on the provided context.\n\n---\n\nFollow the following format.\n\nContext: facts here are assumed to be true\n\nText: ${text}\n\nFaithfulness: True/False indicating if text is faithful to context\n\n---\n\nContext: The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page.\n\nText: Lee scored 3 goals for Colchester United.\n\nFaithfulness:",
    },
    {
        "signature": COTCheckCitationFaithfulness,
        "output": "produce the faithfulness. We know that Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. However, there is no mention of him scoring three goals for Colchester United.\n\nFaithfulness: False",
        "input_kwargs": {
            "context": [
                "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
            ],
            "text": "Lee scored 3 goals for Colchester United.",
        },
        "output_kwargs": {
            "rationale": "produce the faithfulness. We know that Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. However, there is no mention of him scoring three goals for Colchester United.",
            "faithfulness": "False",
        },
        "prompt": "Verify that the text is based on the provided context.\n\n---\n\nFollow the following format.\n\nContext: facts here are assumed to be true\n\nText: ${text}\n\nRationale: Think step by step in order to generate the faithfulness.\n\nFaithfulness: True/False indicating if text is faithful to context\n\n---\n\nContext: The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page.\n\nText: Lee scored 3 goals for Colchester United.\n\nRationale:",
    },
    {
        "signature": COTCheckCitationFaithfulness,
        "output": "produce the faithfulness. We know that Lee had two loan spells in League One last term.",
        "input_kwargs": {
            "context": ["The 21-year-old made seven appearances for the Hammers."],
            "text": "Lee scored 3 goals for Colchester United.",
        },
        "output_kwargs": {
            "rationale": "produce the faithfulness. We know that Lee had two loan spells in League One last term.",
        },
        "prompt": "Verify that the text is based on the provided context.\n\n---\n\nFollow the following format.\n\nContext: facts here are assumed to be true\n\nText: ${text}\n\nRationale: Think step by step in order to generate the faithfulness.\n\nFaithfulness: True/False indicating if text is faithful to context\n\n---\n\nContext: The 21-year-old made seven appearances for the Hammers.\n\nText: Lee scored 3 goals for Colchester United.\n\nRationale:",
    },
]


def test_example_initialization():
    for scenario in TEMPLATE_SCENARIOS:
        template = TextTemplate()
        example = Example(scenario["signature"], **scenario["input_kwargs"])
        assert template.generate(scenario["signature"], example) == scenario["prompt"], print(template.generate(scenario["signature"], example))


def test_template_extraction():
    for scenario in TEMPLATE_SCENARIOS:
        template = TextTemplate()
        example = Example(scenario["signature"], **scenario["input_kwargs"])
        extracted = template.extract(scenario["signature"], example, scenario["output"])
        correct_example = Example(
            **scenario["input_kwargs"], **scenario["output_kwargs"]
        )

        assert extracted == correct_example
