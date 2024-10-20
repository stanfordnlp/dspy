import os
from datasets import load_dataset
from dspy.teleprompt.pez import BootstrapFewShotWithPEZ
from dspy.teleprompt.finetune import BootstrapFinetune

# Load the GLUE SST-2 dataset
dataset = load_dataset("glue", "sst2")
trainset = dataset['train']  # Use the training set for few-shot bootstrapping

# Initialize the PEZ-based few-shot optimizer
fewshot_optimizer = BootstrapFewShotWithPEZ(
    metric=lambda gold, prediction, trace: gold['label'] == prediction['label'],  # A simple metric comparing labels
    max_bootstrapped_demos=4,  # Number of bootstrapped examples
    max_labeled_demos=16,  # Maximum number of labeled demos
    num_candidate_programs=8,  # Number of candidate programs to test
    prompt_len=5,  # Number of tokens in the prompt
    opt_iters=500,  # Number of optimization iterations
    lr=5e-5,  # Learning rate for optimization
    weight_decay=1e-4,  # Weight decay for optimization
    print_step=50,  # Print optimization status every 50 steps
    loss_weight=1.0  # Loss weight during optimization
)

# Instantiate the finetuning model (BootstrapFinetune)
finetune_student = BootstrapFinetune()

# Compile the student model with few-shot optimization via PEZ
teacher_model = "roberta-large"  # Using RoBERTa-large for both prompt optimization and fine-tuning
compiled_program = fewshot_optimizer.compile(
    student=finetune_student,
    teacher=teacher_model,
    trainset=trainset
)

# Now `compiled_program` contains the student model with optimized few-shot prompts
print("Few-shot optimization and finetuning completed.")
import ipdb
ipdb.set_trace()
