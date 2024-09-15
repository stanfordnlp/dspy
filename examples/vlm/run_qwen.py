import dspy
from dspy.datasets import DataLoader
from dspy.adapters.vllm_adapter import encode_image
from dspy.teleprompt import LabeledFewShot

#TODO: Move into example notebook
lm = dspy.VLLMLM(model="Qwen/Qwen2-VL-7B-Instruct", port=8000, url="http://localhost")

adapter = dspy.ImageChatAdapter()
dspy.settings.configure(adapter=adapter, lm=lm)

predictor = dspy.Predict("question, image_1, image_2, image_3, image_4, image_5, image_6, image_7 -> answer", temperature=0.0)
# print(predictor(question="What is the capital of france?"))

input_keys = tuple([f"image_{i}" for i in range(1, 8)] + ["question"])
dataset = DataLoader().from_huggingface("MMMU/MMMU", "Math", split=["dev", "validation"], input_keys=input_keys)

sample_input = dataset["dev"][0]

optimizer = LabeledFewShot(k=2)
bootstrapped_predictor = optimizer.compile(predictor, trainset=dataset["dev"][1:])
print(sample_input.inputs())
# print(encode_image(sample_input.inputs()["image_1"]))
# print(sample_input.inpi)
print(bootstrapped_predictor(**sample_input.inputs()))