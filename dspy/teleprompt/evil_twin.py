import torch
import dspy
from dspy.teleprompt.teleprompt import Teleprompter
import json
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from dspy.teleprompt.utils import get_signature, set_signature

# =============================================================================
# Official helper functions (copied and annotated)
# =============================================================================

def compute_neg_log_prob(model, seq: torch.Tensor, pred_slice: slice, target_slice: slice) -> torch.Tensor:
    pred_logits = model(seq).logits[:, pred_slice, :]
    log_probs = -F.cross_entropy(
        rearrange(pred_logits, "b k v -> b v k"),  # rearrange so that vocab dim is second
        seq[:, target_slice],                      # true target tokens
        reduction="none"
    )
    return log_probs

def compute_grads(model, seq: torch.Tensor, prompt_slice: slice, doc_slice: slice, gamma: float = 0.0) -> torch.Tensor:
    model_embs = model.get_input_embeddings().weight  # shape: (vocab_size, emb_dim)
    one_hot_suffix = torch.zeros(
        seq.shape[0],
        prompt_slice.stop - prompt_slice.start,
        model_embs.shape[0],
        device=model.device,
        dtype=model_embs.dtype,
    )
    # Scatter the one-hot representation for each token in the prompt
    one_hot_suffix.scatter_(-1, rearrange(seq[:, prompt_slice], "b k -> b k 1"), 1)
    one_hot_suffix.requires_grad = True

    # Project to embedding space: (batch, prompt_length, emb_dim)
    suffix_embs = one_hot_suffix @ model_embs
    # Get original embeddings (detach to avoid interfering with gradients)
    embs = model.get_input_embeddings()(seq).detach()
    # Replace the prompt region with the differentiable version while leaving the document embeddings unchanged.
    full_embs = torch.cat(
        [
            embs[:, : prompt_slice.start, :],
            suffix_embs,
            embs[:, prompt_slice.stop :, :],
        ],
        dim=1,
    )

    # Forward pass with modified embeddings.
    logits = model(inputs_embeds=full_embs).logits
    # Targets for the document region.
    targets = seq[:, doc_slice]
    # loss_slice is the region for which predictions are made for the document (shifted by one)
    loss_slice = slice(doc_slice.start - 1, doc_slice.stop - 1)

    # Fluency penalty on the prompt: compute loss over the prompt region
    prompt_pred_slice = slice(prompt_slice.start, prompt_slice.stop - 1)
    prompt_target_slice = slice(prompt_pred_slice.start + 1, prompt_pred_slice.stop + 1)
    # Compute cross entropy over entire batch; sum token losses per sample, then average across batch.
    fluency_loss = F.cross_entropy(
        rearrange(logits[:, prompt_pred_slice, :], "b k v -> b v k"),
        seq[:, prompt_target_slice],
        reduction="none",
    ).sum(dim=-1)
    fluency_penalty = gamma * fluency_loss.mean()

    loss = F.cross_entropy(
        rearrange(logits[:, loss_slice, :], "b k v -> b v k"), targets
    )
    loss += fluency_penalty
    loss.backward()
    # Return the averaged gradient for each token in the prompt region.
    return one_hot_suffix.grad.clone().mean(dim=0)

# =============================================================================
# Main EvilTwin Optimizer Implementation (GCG-based)
# =============================================================================

class LocalModelManager:
    """
    Initializes and manages the local language model and tokenizer.
    
    Provides utility functions for tokenization and computing logits.
    """
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure the tokenizer has a padding token for batching.
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def tokenize(self, text):
        """Tokenizes input text into a list of token IDs, including special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def compute_logits(self, tokens):
        """
        Computes logits for a given list of token IDs.
        
        Args:
          tokens: List of token IDs.
          
        Returns:
          Tensor of shape (1, len(tokens), vocab_size).
        """
        inputs = torch.tensor(tokens, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.logits

class EvilTwin(Teleprompter):
    """
    Implements the Evil Twin optimizer using the Greedy Coordinate Gradient (GCG) algorithm.

    The optimizer finds a hard prompt p that minimizes the KL divergence between the output distribution 
    induced by p and the ground truth prompt p*. It does so by solving the maximum-likelihood problem:

      p* = argmax_p (1/n) ∑_{i=1}^n log PLLM(di | p)
      
    which is equivalent (up to constants) to minimizing the KL divergence 
    d_KL(p* || p) ≈ ∑_i exp(log PLLM(di|p*)) (log PLLM(di|p*) - log PLLM(di|p)).

    This class uses iterative token replacement guided by per-token gradients computed using both the prompt 
    and document context. The process closely follows the GCG algorithm described in the paper.
    """
    def __init__(
        self,
        prompt: str,
        initial_et_prompt = "I have a very important question for you to ask right now.",
        n_epochs=500,
        batch_size=10,
        top_k=256,
        log_fpath="gcg_log.json",
        early_stop_kl=0.0,
        local_model_name="EleutherAI/gpt-neo-125M",
        gamma=0.0,
    ):
        super().__init__()
        self.prompt = prompt
        self.initial_et_prompt = initial_et_prompt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.top_k = top_k
        self.log_fpath = log_fpath
        self.early_stop_kl = early_stop_kl
        self.gamma = gamma  # Fluency penalty coefficient
        self.local_model = LocalModelManager(local_model_name)
        # Will hold pre-tokenized document tokens from original outputs.
        self.original_doc_tokens = None

    def compile(self, program: dspy.Module):
        """
        Compiles the DSPy program by:
          1. Running the original prompt to sample a set of documents.
          2. Computing and storing log probabilities of these documents under the original prompt.
          3. Pre-tokenizing the document texts for efficient KL divergence computation.
          4. Running the GCG optimization to produce an optimized (evil twin) prompt.
        """
        lm = dspy.settings.lm
        if lm is None:
            raise ValueError("DSPy model is not configured. Use `dspy.configure(lm=your_model)`.")
        self.tokenizer = self.local_model.tokenizer
        predictor = program.predictors()[0]
        optim_prompt = self.initial_et_prompt
        # Run the program to sample documents (n_samples = 100).
        original_outputs: list[dspy.Prediction] = self.run_program(program, self.prompt, num_samples=100)
        # Compute original log probabilities for each document: log PLLM(di | p*).
        self.original_log_probs = [
            self.compute_log_prob(self.prompt, doc.answer) for doc in original_outputs
        ]
        # Pre-tokenize documents once for efficient batched evaluation.
        self.original_doc_tokens = [
            self.local_model.tokenizer(
                doc.answer, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
            for doc in original_outputs
        ]

        print(original_outputs[:10])
        best_prompt = self.optim_gcg(optim_prompt)
        updated_signature = get_signature(predictor).with_instructions(best_prompt)
        set_signature(predictor, updated_signature)
        return program

    def run_program(self, program, prompt, num_samples=1):
        """
        Runs the DSPy program with the given prompt.
        
        If num_samples > 1, constructs a batch of examples.
        """
        if num_samples > 1:
            examples = [dspy.Example(question=prompt).with_inputs("question") for _ in range(num_samples)]
            responses = program.batch(examples)
            return responses
        return program(question=prompt)

    def optim_gcg(self, optim_prompt):
        """
        Runs the GCG optimization for a fixed number of epochs.
        
        For each epoch:
          - Computes the candidate KL divergence loss for the current prompt.
          - Attempts token replacements by considering the top-k candidate tokens (based on gradient information).
          - Selects the candidate prompt with the lowest KL divergence.
          - Logs progress and stops early if the loss falls below a threshold.
        """
        print(f"Starting Evil Twin Optimization for: {self.prompt}")
        best_prompt = optim_prompt
        best_loss = float("inf")
        history = []
        pbar = tqdm(range(1, self.n_epochs + 1))
        for epoch in pbar:
            candidate_loss = self.compute_kl_divergence(best_prompt)
            history.append({"epoch": epoch, "loss": candidate_loss, "best_prompt": best_prompt})
            with open(self.log_fpath, "w") as f:
                json.dump(history, f, indent=4)
            pbar.set_description(f"Epoch {epoch}, KL Loss: {candidate_loss:.4f}, Best prompt: {best_prompt}")
            if candidate_loss < self.early_stop_kl:
                print(f"Early stopping: KL loss below {self.early_stop_kl}")
                break
            result = self.replace_tok(best_prompt)
            if result is None:
                continue
            new_prompt, new_loss = result
            if new_loss < best_loss:
                best_loss = new_loss
                best_prompt = new_prompt
        return best_prompt

    def replace_tok(self, prompt):
        """
        Iterates over each token in the prompt and attempts to replace it with one of the top-k candidates.

        Evaluates candidate replacements via KL divergence.
        
        Returns:
          A tuple of (new_prompt, corresponding KL divergence loss).
        """
        tokens = self.local_model.tokenize(prompt)
        best_prompt = prompt
        best_loss = float("inf")
        # Compute per-token gradients (including document context).
        grads = self.compute_gradients(tokens)
        # Get indices of top-k candidate tokens (by largest negative gradient).
        _, top_k_indices = torch.topk(-grads, self.top_k, dim=-1)
        for i in range(len(tokens)):
            # Randomly sample one candidate token from the top-k for position i.
            new_token_idx = random.choice(top_k_indices[i].tolist())
            modified_tokens = tokens[:i] + [new_token_idx] + tokens[i+1:]
            new_prompt = self.tokenizer.decode(modified_tokens)
            # Evaluate candidate prompt's KL divergence.
            candidate_loss = self.compute_kl_divergence(new_prompt)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_prompt = new_prompt
        return best_prompt, best_loss

    def compute_gradients(self, tokens):
        """
        Computes gradients for each token in the prompt by concatenating the prompt with a document.

        To reflect document context (per GCG), the prompt tokens are concatenated with one pre-tokenized document 
        (here, the first one from self.original_doc_tokens). This forms a sequence: [prompt; document].
        Gradients are then computed with respect to the prompt region using the official helper.

        Returns:
          Tensor of shape (n_prompt_tokens, vocab_size) with averaged gradients.
        """
        device = self.local_model.device
        # Convert prompt to tensor and get its length.
        prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, L_prompt)
        L_prompt = prompt_tensor.size(1)
        # Pad all original documents into a batch.
        doc_batch = self.local_model.tokenizer.pad(self.original_doc_tokens, return_tensors="pt")
        # Remove extra dimensions if any.
        if doc_batch['input_ids'].dim() == 3:
            doc_batch['input_ids'] = doc_batch['input_ids'].squeeze(1)
            doc_batch['attention_mask'] = doc_batch['attention_mask'].squeeze(1)
        for key in doc_batch:
            doc_batch[key] = doc_batch[key].to(device)
        # Number of documents.
        num_docs = doc_batch["input_ids"].shape[0]
        # Repeat the prompt for every document.
        prompt_tensor_rep = prompt_tensor.repeat(num_docs, 1)
        # Concatenate the repeated prompt with the batch of documents.
        seq = torch.cat([prompt_tensor_rep, doc_batch["input_ids"]], dim=1)
        prompt_slice = slice(0, L_prompt)
        doc_slice = slice(L_prompt, seq.size(1))
        # Compute gradients over the entire batch and average them.
        return compute_grads(self.local_model.model, seq, prompt_slice, doc_slice, self.gamma)


    def compute_log_prob_from_tokens(self, prompt_tokens, document_tokens):
        """
        Computes the log probability of a document given a candidate prompt, using pre-tokenized inputs.

        Args:
          prompt_tokens: Dictionary from tokenizer for the candidate prompt (2D tensor of shape [1, L_prompt]).
          document_tokens: Dictionary from tokenizer for the document (2D tensor of shape [1, L_doc]).

        Process:
          - Concatenates prompt tokens and document tokens along the sequence dimension.
          - Concatenates attention masks similarly to inform the model which tokens to attend to.
          - A forward pass computes logits over the concatenated sequence.
          - Only the document region is used to compute the cross-entropy loss.
          - The negative loss (summed over the document region) is returned as the log probability.
        
        Slices:
          - pred_slice: slice(L_prompt + 1, T - 1), where T is total length.
          - target_slice: slice(L_prompt + 2, T).
        """
        L_prompt = prompt_tokens.input_ids.shape[1]
        inputs = {
            'input_ids': torch.cat([prompt_tokens.input_ids, document_tokens.input_ids], dim=1),
            'attention_mask': torch.cat([prompt_tokens.attention_mask, document_tokens.attention_mask], dim=1),
        }
        inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.local_model.model(**inputs).logits
        T = inputs['input_ids'].shape[1]
        pred_slice = slice(L_prompt + 1, T - 1)
        target_slice = slice(L_prompt + 2, T)
        neg_log_prob = F.cross_entropy(
            logits[:, pred_slice, :].reshape(-1, logits.shape[-1]),
            inputs['input_ids'][:, target_slice].reshape(-1),
            reduction='mean'
        )
        return -neg_log_prob.item()

    def compute_log_prob(self, prompt, document):
        """
        Tokenizes a prompt and a document, then computes the log probability of the document given the prompt.
        """
        prompt_tokens = self.local_model.tokenizer(prompt, return_tensors="pt", truncation=True)
        document_tokens = self.local_model.tokenizer(document, return_tensors="pt", truncation=True)
        return self.compute_log_prob_from_tokens(prompt_tokens, document_tokens)

    def compute_kl_divergence(self, candidate_prompt):
        """
        Computes the approximate KL divergence between the original prompt and a candidate prompt.
        
        KL divergence is approximated as:
          KL ≈ ∑_i exp(log_prob_orig_i) * (log_prob_orig_i - log_prob_candidate_i)
        where log_prob_orig_i is the log probability (under the original prompt) for document i,
        and log_prob_candidate_i is the log probability for document i under the candidate prompt.

        Batching:
          - Tokenizes the candidate prompt once.
          - Pads the pre-tokenized original documents into a batch (ensuring 2D tensors).
          - Repeats the candidate prompt to match the batch size.
          - Concatenates candidate prompt tokens with document tokens.
          - Runs a single forward pass to obtain logits for all documents.
          - Computes the summed negative log probability per document over the document region.
        
        Returns:
          Mean KL divergence over the batch.
        """
        candidate_prompt_tokens = self.local_model.tokenizer(candidate_prompt, return_tensors="pt", truncation=True)
        # Ensure candidate prompt tensors are 2D.
        cand_ids = candidate_prompt_tokens.input_ids
        if cand_ids.dim() > 2:
            cand_ids = cand_ids.squeeze(0)
        cand_mask = candidate_prompt_tokens.attention_mask
        if cand_mask.dim() > 2:
            cand_mask = cand_mask.squeeze(0)
        candidate_prompt_tokens = {"input_ids": cand_ids, "attention_mask": cand_mask}
        # Pad the list of pre-tokenized documents into a batch.
        batch = self.local_model.tokenizer.pad(self.original_doc_tokens, return_tensors="pt")
        if batch['input_ids'].dim() == 3:
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)

        batch_size = batch['input_ids'].size(0)
        # Repeat candidate prompt tokens to match the batch size.
        candidate_prompt_batch = {
            'input_ids': candidate_prompt_tokens["input_ids"].repeat(batch_size, 1),
            'attention_mask': candidate_prompt_tokens["attention_mask"].repeat(batch_size, 1)
        }
        # Concatenate candidate prompt with document tokens.
        inputs = {
            'input_ids': torch.cat([candidate_prompt_batch['input_ids'], batch['input_ids']], dim=1),
            'attention_mask': torch.cat([candidate_prompt_batch['attention_mask'], batch['attention_mask']], dim=1)
        }
        inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.local_model.model(**inputs).logits
        T = inputs['input_ids'].size(1)
        L_prompt = candidate_prompt_tokens["input_ids"].size(1)
        pred_slice = slice(L_prompt + 1, T - 1)
        target_slice = slice(L_prompt + 2, T)
        # Compute negative log probabilities for each document.
        neg_log_probs = F.cross_entropy(
            logits[:, pred_slice, :].reshape(-1, logits.shape[-1]),
            inputs['input_ids'][:, target_slice].reshape(-1),
            reduction='none'
        )
        neg_log_probs = neg_log_probs.view(batch_size, -1).mean(dim=1)
        candidate_log_probs = -neg_log_probs.cpu()
        original_log_probs = torch.tensor(self.original_log_probs)
        # Compute the weighted KL divergence.
        kl_div = torch.sum(torch.exp(original_log_probs) * (original_log_probs - candidate_log_probs))
        return kl_div.mean().item()
