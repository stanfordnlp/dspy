import dspy
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

class ProbabilisticClassificationSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField(desc="An image to classify")
    class_labels: List[str] = dspy.InputField(desc="Possible class labels")
    probabilities: Dict[str, float] = dspy.OutputField(desc="Probability distribution over the class labels")

example_inputs_classification = [
    dspy.Example(image=dspy.Image.from_url("https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg"), class_labels=["cat", "dog", "bird"]).with_inputs("image", "class_labels"),
    dspy.Example(image=dspy.Image.from_url("https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg"), class_labels=["dog", "cat", "bird", "horse", "elephant", "monkey", "tiger", "lion", "bear", "zebra", "giraffe"]).with_inputs("image", "class_labels")
]

class ImageToCodeSignature(dspy.Signature):
    ui_image: dspy.Image = dspy.InputField(desc="An image of a user interface")
    target_language: str = dspy.InputField(desc="Programming language for the generated code")
    generated_code: str = dspy.OutputField(desc="Code that replicates the UI shown in the image")

example_inputs_image_to_code = [
    dspy.Example(ui_image=dspy.Image.from_url("https://nikitahl.com/images/button-styles/buttons-chrome.png"), target_language="HTML").with_inputs("ui_image", "target_language"),
    dspy.Example(ui_image=dspy.Image.from_url("https://nikitahl.com/images/button-styles/buttons-chrome.png"), target_language="Swift").with_inputs("ui_image", "target_language"),
    dspy.Example(ui_image=dspy.Image.from_url("https://nikitahl.com/images/button-styles/buttons-chrome.png"), target_language="Spanish").with_inputs("ui_image", "target_language")
]

class BBOXImageSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField(desc="The original image to annotate")
    bboxes: list[tuple[int, int, int, int]] = dspy.OutputField(desc="List of bounding boxes with coordinates (x1, y1, x2, y2)")

example_inputs_bbox_image = [
    dspy.Example(image=dspy.Image.from_url("https://nikitahl.com/images/button-styles/buttons-chrome.png")).with_inputs("image")
]

class MultilingualCaptionSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField(desc="An image to generate captions for")
    languages: list[str] = dspy.InputField(desc="List of language codes for the captions (e.g., ['en', 'es', 'fr'])")
    captions: dict[str, str] = dspy.OutputField(desc="Captions in different languages keyed by language code")

example_inputs_multilingual_caption = [
    dspy.Example(image=dspy.Image.from_url("https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg"), languages=["en", "es", "fr"]).with_inputs("image", "languages"),
]
# dspy.Example(image=dspy.Image.from_url("https://nikitahl.com/images/button-styles/buttons-chrome.png"), languages=["en", "es", "fr", "de", "it", "ja", "zh", "ko"]).with_inputs("image", "languages")

signature_test_cases = {
    ProbabilisticClassificationSignature: example_inputs_classification,
    ImageToCodeSignature: example_inputs_image_to_code,
    BBOXImageSignature: example_inputs_bbox_image,
    MultilingualCaptionSignature: example_inputs_multilingual_caption
}
# Try some nested signatures


# haiku_lm = dspy.LM(model="anthropic/claude-3-haiku-20240307", max_tokens=4096)
# vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct --trust-remote-code --limit-mm-per-prompt image=16 --seed 42 --enforce-eager --max-num-seqs 48
internlm_lm = dspy.LM(model="openai/OpenGVLab/InternVL2-8B", api_base="http://localhost:8000/v1", api_key="sk-fake-key", max_tokens=5000)
gpt_lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=5000)
all_lms = [internlm_lm, gpt_lm]

def test_signature(lm: dspy.LM, signature: dspy.Signature):
    predictor = dspy.Predict(signature)
    with dspy.context(lm=lm):
        for example_inputs in signature_test_cases[signature]:
            print(predictor(**example_inputs.inputs()))

for lm in all_lms:
    for signature in signature_test_cases:
        test_signature(lm, signature)
