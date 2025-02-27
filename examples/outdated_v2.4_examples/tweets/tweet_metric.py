import dspy
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=200, test_size=0, keep_details=True)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.without('id', 'type').with_inputs('question') for x in dataset.train]
devset = [x.without('id', 'type').with_inputs('question') for x in dataset.dev]
valset, devset = devset[:50], devset[50:]


# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc='ignore if N/A')
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')
retrieve = dspy.Retrieve(k=5)
METRIC = None

def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output
    context = retrieve(question).passages

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    correct = f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    correct = f"{correct} Does the assessed text above contain the gold answer?"
    
    with dspy.context(lm=gpt4T):
        faithful = dspy.Predict(Assess)(context=context, assessed_text=tweet, assessment_question=faithful)
        correct =  dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=correct)
        engaging = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=engaging)

    correct, engaging, faithful = (m.assessment_answer.split()[0].lower() == 'yes' for m in [correct, engaging, faithful])
    score = (correct + engaging + faithful) if correct and (len(tweet) <= 280) else 0

    if METRIC is not None:
        if METRIC == 'correct':
            return correct
        if METRIC == 'engaging':
            return engaging
        if METRIC == 'faithful':
            return faithful

    if trace is not None:
        return score >= 3
    return score / 3.0
