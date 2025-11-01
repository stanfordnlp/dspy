import dspy
import arbor
from arbor import ArborGRPO, ArborProvider
arbor_server_info = arbor.init() # Initialize the Arbor server in the background

port = 7453
local_lm_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_lm = dspy.LM(
    model=f"openai/arbor:{local_lm_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
    # Arbor checks to make sure these match the training config
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=2048,
)

dspy.configure(lm=local_lm)


import ujson
import bm25s
import Stemmer

corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = ujson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:600], hover[600:900], hover[900:]
len(trainset), len(devset), len(testset)

def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return list(run.keys())

instr1 = """
Given a claim and some key facts, generate a follow-up search query to find the next most essential clue towards verifying or refuting the claim. The goal ultimately is to find all documents implicated by the claim.
""".strip()

instr2 = """
Given a claim, some key facts, and new search results, identify any new learnings from the new search results, which will extend the key facts known so far about the whether the claim is true or false. The goal is to ultimately collect all facts that would help us find all documents implicated by the claim.
"""


class ResearchHop(dspy.Module):
    def __init__(self, num_docs, num_hops):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(dspy.Signature("claim, key_facts -> followup_search_query", instr1))
        self.append_notes = dspy.ChainOfThought(dspy.Signature("claim, key_facts, new_search_results -> new_key_facts", instr2))

    def forward(self, claim: str) -> list[str]:
        key_facts = []
        retrieved_docs = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(claim=claim, key_facts=key_facts).followup_search_query if hop_idx else claim
            search_results = search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)

            if hop_idx == self.num_hops - 1:
                break
                
            prediction = self.append_notes(claim=claim, key_facts=key_facts, new_search_results=search_results)
            key_facts.append(prediction.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)

def recall(example, pred, trace=None):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)

evaluate = dspy.Evaluate(devset=devset, metric=recall, num_threads=16, display_progress=True, display_table=5)

arbor_server_info = arbor.init()

program = ResearchHop(num_docs=4, num_hops=2)
program.set_lm(local_lm)

# NOTE: Training on 4 GPUs.
train_kwargs = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "beta": 0.00,
    "learning_rate": 1e-6,
    "gradient_checkpointing": True,
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "loss_type": "dapo",
    "max_steps": 1000,
    "report_to": "wandb",
    "log_completions": True,
    "logging_steps": 10,
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": False,
    "max_grad_norm": 1.0,
    "lora_config": {
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    },
    "num_training_gpus": 3,
    "num_inference_gpus": 1,
}

compiler = ArborGRPO(
    metric=recall,
    num_dspy_examples_per_grpo_step=2,
    num_rollouts_per_grpo_step=12,
    exclude_demos=True,
    num_train_steps=1000,
    num_threads=16,
    use_train_as_val=False,
    num_steps_for_val=50,
    train_kwargs=train_kwargs,
    checkpoint="single-best",
)

optimized_program = compiler.compile(
    student=program,
    trainset=trainset,
    valset=devset,
)
