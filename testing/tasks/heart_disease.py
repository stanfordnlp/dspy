import dspy
import random
from datasets import load_dataset
from .base_task import BaseTask

mappings = {
    "sex": {0: "female", 1: "male"},
    "cp": {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal pain",
        4: "asymptomatic",
    },
    "restecg": {
        0: "normal",
        1: "ST-T wave abnormality",
        2: "left ventricular hypertrophy",
    },
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {"3": "normal", "6": "fixed defect", "7": "reversible defect"},
    "target": {0: "no", 1: "yes"},  # presence of heart disease
}

dataset = load_dataset("buio/heart-disease")
fullset = []

for x in dataset["train"]:
    for key, value in x.items():
        if key in mappings:
            x[key] = mappings[key].get(value, value)
        x[key] = str(x[key])

    inputs = [k for k in x.keys() if k != "target"]
    x["answer"] = x["target"]
    del x["target"]

    fullset.append(dspy.Example(**x).with_inputs(*inputs))

random.Random(0).shuffle(fullset)

trainset = fullset[:120]
devset = trainset
testset = fullset[120:]


class HeartDiseaseInput(dspy.Signature):
    age = dspy.InputField(desc="Age in years")
    sex = dspy.InputField(desc="Sex (male or female)")
    cp = dspy.InputField(
        desc="Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)"
    )
    trestbps = dspy.InputField(
        desc="Resting blood pressure (in mm Hg on admission to the hospital)"
    )
    chol = dspy.InputField(desc="Serum cholestoral in mg/dl")
    fbs = dspy.InputField(desc="Fasting blood sugar > 120 mg/dl (true or false)")
    restecg = dspy.InputField(
        desc="Resting electrocardiographic results (normal, ST-T wave abnormality, left ventricular hypertrophy)"
    )
    thalach = dspy.InputField(desc="Maximum heart rate achieved")
    exang = dspy.InputField(desc="Exercise induced angina (yes or no)")
    oldpeak = dspy.InputField(desc="ST depression induced by exercise relative to rest")
    slope = dspy.InputField(
        desc="The slope of the peak exercise ST segment (upsloping, flat, downsloping)"
    )
    ca = dspy.InputField(desc="Number of major vessels (0-3) colored by flourosopy")
    thal = dspy.InputField(desc="Thalassemia (normal, fixed defect, reversible defect)")


class HeartDiseaseSignature(HeartDiseaseInput):
    """Given patient information, predict the presence of heart disease."""

    answer = dspy.OutputField(
        desc="Does this patient have heart disease? Just yes or no."
    )


class HeartDiseaseVote(HeartDiseaseInput):
    """Given patient information, predict the presence of heart disease. I can critically assess the provided trainee opinions."""

    context = dspy.InputField(desc="A list of opinions from trainee doctors.")
    answer = dspy.OutputField(
        desc="Does this patient have heart disease? Just yes or no."
    )


class Classify(dspy.Module):
    def __init__(self):
        self.classify = [
            dspy.ChainOfThought(HeartDiseaseSignature, temperature=0.7 + i * 0.01)
            for i in range(3)
        ]
        self.vote = dspy.ChainOfThought(HeartDiseaseVote)

    def forward(
        self,
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ):
        kwargs = dict(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal,
        )

        opinions = [c(**kwargs) for c in self.classify]
        opinions = [
            (opinion.reasoning.replace("\n", " ").strip("."), opinion.answer.strip("."))
            for opinion in opinions
        ]

        opinions = [
            f"I'm a trainee doctor, trying to {reason}. Hence, my answer is {answer}."
            for reason, answer in opinions
        ]
        return self.vote(context=opinions, **kwargs)


class HeartDiseaseTask(BaseTask):
    def __init__(self):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.metric = dspy.evaluate.answer_exact_match

    def get_program(self):
        return Classify()

    def get_metric(self):
        return self.metric

    def get_default_max_bootstrapped_demos(self):
        return 4

    def get_default_max_labeled_demos(self):
        return 0

    def get_trainset(self, TRAIN_NUM=None):
        return self.trainset[:TRAIN_NUM]

    def get_testset(self, TEST_NUM=None):
        return self.testset[:TEST_NUM]

    def get_devset(self, TRAIN_NUM=None, DEV_NUM=None):
        return self.devset[:TRAIN_NUM]

    def get_handwritten_prompts(self):
        return []

    def get_max_tokens(self):
        return 350
