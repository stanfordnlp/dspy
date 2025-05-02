import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset

# Load the Banking77 dataset.
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

# Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
]

random.Random(0).shuffle(raw_data)
print(len(CLASSES), CLASSES[:10])

trainset = raw_data[:1500] # 1500 examples for training
valset = raw_data[1500:1600] # 100 examples for validation
print(trainset[0])

classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")

from dspy.clients.lm_local_arbor import ArborProvider
port = 7453
arbor_api_base = f"http://localhost:{port}/v1/"
api_key = "arbor"
provider = ArborProvider()

# student_lm_name = "meta-llama/Llama-3.2-3B-Instruct"
# student_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_lm_name = "Qwen/Qwen3-0.6B"
student_lm = dspy.LM(model=f"openai/arbor:{student_lm_name}", provider=provider, temperature=0.5, api_base=arbor_api_base, api_key=api_key, max_tokens=3000)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

metric = (lambda x, y, trace=None: x.label == y.label)


from dspy.teleprompt.grpo import GRPO
train_kwargs = {
    "update_interval": 10,
    "per_device_train_batch_size": 4,
    "temperature": 0.5,
    "beta": 0.02,
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
}

compiler = GRPO(
    metric=metric,
    multitask=True,
    num_dspy_examples_per_grpo_step=4,
    num_samples_per_input=4,
    exclude_demos=True,
    num_train_steps=500,
    num_threads=4,
    use_train_as_val=False,
    num_steps_for_val=100,
    train_kwargs=train_kwargs,
)

classify_ft = compiler.compile(
    student=student_classify,
    trainset=trainset,
    valset=valset,
)

# evaluate = dspy.Evaluate(devset=valset, metric=metric, display_progress=True, display_table=5, num_threads=16)
# print(evaluate(classify_ft))
