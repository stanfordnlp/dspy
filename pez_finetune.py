import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset, load_metric
from dspy.teleprompt import PEZFewshot, PEZFinetune

# Step 1: Load the GLUE dataset (SST-2)
def load_sst2_dataset():
    dataset = load_dataset("glue", "sst2")
    metric = load_metric("glue", "sst2")
    return dataset, metric

# Step 2: Initialize tokenizer and model
def initialize_roberta_model():
    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Step 3: Define a metric to evaluate during few-shot and fine-tuning
def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = len(labels)
    return correct / total

# Step 4: Compile and optimize prompts with PEZFewshot
def run_fewshot_training(model, tokenizer, dataset, trainset_size=16):
    # Get a few-shot training set
    trainset = dataset['train'].select(range(trainset_size))
    valset = dataset['validation']

    # Define the metric function
    metric = lambda example, pred, trace: compute_accuracy(pred.argmax(dim=1).cpu(), example['label'])

    # Initialize the PEZFewshot teleprompter
    fewshot_optimizer = PEZFewshot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=trainset_size,
        prompt_len=5,
        iter=500,
        lr=5e-5,
        weight_decay=1e-4,
        print_step=50,
    )

    # Compile the model and optimize the prompt using few-shot examples
    compiled_model = fewshot_optimizer.compile(model, trainset=trainset)

    return compiled_model, valset

# Step 5: Fine-tune the model with PEZFinetune
def run_finetuning(compiled_model, dataset):
    trainset = dataset['train']
    valset = dataset['validation']

    # Define the metric function
    metric = lambda example, pred, trace: compute_accuracy(pred.argmax(dim=1).cpu(), example['label'])

    # Initialize the PEZFinetune teleprompter
    finetune_optimizer = PEZFinetune(metric=metric)

    # Compile and fine-tune the model with optimized prompts
    finetuned_model = finetune_optimizer.compile(compiled_model, trainset=trainset, valset=valset, target="roberta-large")

    return finetuned_model, valset

# Step 6: Evaluate the fine-tuned model
def evaluate_model(model, tokenizer, dataset, valset):
    model.eval()

    # Tokenize the validation set
    val_encodings = tokenizer(valset['sentence'], truncation=True, padding=True, return_tensors='pt')

    # Run predictions
    with torch.no_grad():
        inputs = {key: val_encodings[key].to(model.device) for key in val_encodings}
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1).cpu()

    # Calculate accuracy
    accuracy = compute_accuracy(preds, valset['label'])
    print(f"Validation Accuracy: {accuracy:.4f}")

# Step 7: Main script to run the experiment
def main():
    # Load dataset and model
    dataset, metric = load_sst2_dataset()
    model, tokenizer = initialize_roberta_model()

    # Run PEZFewshot to generate optimized prompts
    compiled_model, valset = run_fewshot_training(model, tokenizer, dataset)

    # Fine-tune the model using the optimized prompts
    finetuned_model, valset = run_finetuning(compiled_model, dataset)

    # Evaluate the fine-tuned model
    evaluate_model(finetuned_model, tokenizer, dataset, valset)

if __name__ == "__main__":
    main()
