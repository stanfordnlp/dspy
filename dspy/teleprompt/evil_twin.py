import dspy
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from dspy.teleprompt.teleprompt import Teleprompter


class EvilTwin(Teleprompter):
    def __init__(
        self,
        n_epochs=500,
        kl_every=1,
        batch_size=10,
        top_k=256,
        gamma=0.0,
        log_fpath="gcg_log.json",
        early_stop_kl=0.0,
    ):
        """
        DSPy-compatible optimizer that refines a hard prompt using Greedy Coordinate Gradient (GCG).

        Parameters:
        - n_epochs: Number of epochs for GCG.
        - kl_every: Frequency of KL divergence evaluation.
        - batch_size: Batch size for optimization.
        - top_k: Number of top prompts retained at each step.
        - gamma: Regularization factor for fluency penalty.
        - log_fpath: File to save optimization logs.
        - early_stop_kl: If KL goes below this, stop optimization.
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.kl_every = kl_every
        self.batch_size = batch_size
        self.top_k = top_k
        self.gamma = gamma
        self.log_fpath = log_fpath
        self.early_stop_kl = early_stop_kl

    def compile(self, module: dspy.Module):
        """
        Optimizes a DSPy module using GCG to generate an "evil twin" prompt.

        Parameters:
        - module: The DSPy module to optimize.

        Returns:
        - Optimized module with an updated "evil twin" prompt.
        """
        lm = dspy.settings.lm
        if lm is None:
            raise ValueError("DSPy model is not configured. Use `dspy.configure(lm=your_model)`.")  

        # Extract prompt
        if not hasattr(module, "predict") or not hasattr(module.predict, "signature"):
            raise ValueError("Module does not have a valid `predict` signature.")
        
        original_prompt = module.predict.signature.instructions

        # Initialize adversarial prompt
        optim_prompt = "! " * 15  

        # Run GCG optimization
        best_prompt = self.optim_gcg(
            lm=lm, 
            original_prompt=original_prompt, 
            optim_prompt=optim_prompt
        )

        # Update module with optimized prompt
        module.predict.signature.instructions = best_prompt
        return module

    def optim_gcg(self, lm, original_prompt, optim_prompt):
        """
        Optimizes a prompt using Greedy Coordinate Gradient (GCG).

        Parameters:
        - lm: DSPy language model.
        - original_prompt: Original text prompt.
        - optim_prompt: Initial adversarial prompt.

        Returns:
        - Optimized adversarial prompt.
        """
        print(f"Starting Evil Twin Optimization for: {original_prompt}")

        best_prompt = optim_prompt
        best_loss = float("inf")
        history = []

        pbar = tqdm(range(1, self.n_epochs + 1))
        for epoch in pbar:
            # Generate outputs for both original and adversarial prompts
            original_output = lm.generate(original_prompt, n=self.batch_size)
            adversarial_output = lm.generate(best_prompt, n=self.batch_size)

            # Compute KL divergence loss
            loss = self.compute_kl_divergence(original_output, adversarial_output)

            # Generate modified prompts by replacing individual tokens
            new_prompt, new_loss = self.replace_tokens(lm, best_prompt)

            # Update best prompt if loss improved
            if new_loss < best_loss:
                best_loss = new_loss
                best_prompt = new_prompt

            # Log progress
            history.append({"epoch": epoch, "loss": loss, "best_prompt": best_prompt})
            with open(self.log_fpath, "w") as f:
                json.dump(history, f, indent=4)

            pbar.set_description(f"Epoch {epoch}, KL Loss: {loss:.4f}")

            if loss < self.early_stop_kl:
                print(f"Early stopping: KL loss below {self.early_stop_kl}")
                break

        return best_prompt

    def replace_tokens(self, lm, prompt):
        """
        Iteratively replaces each token in the prompt, computing loss and keeping the best replacement.

        Parameters:
        - lm: DSPy language model.
        - prompt: Current adversarial prompt.

        Returns:
        - Optimized prompt with the best token replacements.
        - Loss value after optimization.
        """
        tokens = lm.tokenize(prompt)
        best_prompt = prompt
        best_loss = float("inf")

        for i in range(len(tokens)):
            # Generate candidates by replacing one token at a time
            for new_token in self.get_top_k_replacements(lm, tokens[i]):
                modified_tokens = tokens[:i] + [new_token] + tokens[i + 1:]
                new_prompt = lm.detokenize(modified_tokens)

                # Compute KL divergence loss
                new_output = lm.generate(new_prompt, n=self.batch_size)
                new_loss = self.compute_kl_divergence(lm.generate(prompt, n=self.batch_size), new_output)

                # Keep best replacement
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_prompt = new_prompt

        return best_prompt, best_loss

    def get_top_k_replacements(self, lm, token):
        """
        Retrieves the top-k alternative tokens for a given token based on model predictions.

        Parameters:
        - lm: DSPy language model.
        - token: Token to be replaced.

        Returns:
        - List of top-k alternative tokens.
        """
        logits = lm.get_token_logits(token)
        top_k_indices = torch.topk(logits, self.top_k).indices
        return [lm.detokenize([idx]) for idx in top_k_indices]

    def compute_kl_divergence(self, original_output, adversarial_output):
        """
        Computes KL divergence between original and adversarial prompt outputs using Torch.

        Returns:
        - KL divergence score (lower is better).
        """
        # Convert outputs to probability distributions
        original_probs = torch.tensor(original_output, dtype=torch.float32)
        adversarial_probs = torch.tensor(adversarial_output, dtype=torch.float32)

        original_probs /= original_probs.sum(dim=-1, keepdim=True)
        adversarial_probs /= adversarial_probs.sum(dim=-1, keepdim=True)

        # Compute KL divergence
        kl_div = torch.sum(original_probs * torch.log(original_probs / (adversarial_probs + 1e-8)), dim=-1)

        return kl_div.mean().item()
