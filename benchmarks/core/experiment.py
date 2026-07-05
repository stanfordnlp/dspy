"""
Main experiment runner for prompt optimization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy

from core.config import ExperimentConfig
from core.logging import ExperimentLogger
from data_adapters.registry import DatasetRegistry
from optimizers.registry import OptimizerRegistry
from programs.registry import ProgramRegistry

if TYPE_CHECKING:
    from dspy import Module

# Try to import cloudpickle, fall back to pickle if not available
try:
    import cloudpickle as pickle
except ImportError:
    import pickle


class ExperimentRunner:
    """Main runner for prompt optimization experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(
            experiment_name=config.name,
            log_dir=config.logging.log_dir,
            level=config.logging.level,
            log_to_file=config.logging.log_to_file,
            include_timestamps=config.logging.include_timestamps
        )
        
        # Log the configuration
        self.logger.log_config(self._config_to_dict(config))
        
    def run(self) -> dict[str, Any]:
        """Run the complete experiment.
        
        Returns:
            Dictionary containing experiment results.
        """
        # Setup model
        self._setup_model()
        
        # Load dataset
        dataset_adapter = DatasetRegistry.create_adapter(
            self.config.dataset.name, 
            self._dataset_config_to_dict()
        )
        train_set, val_set, test_set = dataset_adapter.load_dataset()
        
        self.logger.log_dataset_loaded(
            dataset_adapter.name,
            len(train_set),
            len(val_set), 
            len(test_set)
        )
        final_eval_set = test_set if test_set else val_set
        final_eval_split = "test" if test_set else "validation"
        if test_set:
            self.logger.logger.info(
                f"Final reporting will use held-out test set ({len(test_set)} examples); "
                f"optimization will use validation set ({len(val_set)} examples)."
            )
        else:
            self.logger.logger.info(
                "No test set configured; final reporting will fall back to validation set."
            )
        self.logger._add_timeline_event(
            "final_eval_split_selected",
            {"split": final_eval_split, "size": len(final_eval_set)}
        )
        
        # Create program
        program = ProgramRegistry.create_program(
            self.config.program.name,
            self.config.program.params
        )
        
        # Evaluate baseline
        self.logger.logger.info("=" * 60)
        self.logger.logger.info("BASELINE EVALUATION")
        self.logger.logger.info("=" * 60)
        
        metric = dataset_adapter.get_metric()
        baseline_results = self._evaluate_program(
            program,
            final_eval_set,
            metric,
            f"Baseline ({final_eval_split})",
        )
        self.logger.log_baseline_results(baseline_results, len(final_eval_set))
        
        # Run optimization (if not baseline)
        optimized_results = None
        optimized_program = None
        
        if self.config.optimizer.name != "baseline":
            self.logger.logger.info("=" * 60)
            self.logger.logger.info(f"OPTIMIZATION: {self.config.optimizer.name.upper()}")
            self.logger.logger.info("=" * 60)
            
            # Get optimizer
            optimizer_adapter = OptimizerRegistry.create_adapter(
                self.config.optimizer.name,
                self._optimizer_config_to_dict()
            )
            
            # Choose appropriate metric
            optimization_metric = (
                dataset_adapter.get_gepa_metric() 
                if optimizer_adapter.supports_gepa_metric() 
                else dataset_adapter.get_metric()
            )
            
            self.logger.log_optimization_start(optimizer_adapter.name)
            start_time = time.time()
            
            # Run optimization
            optimized_program = optimizer_adapter.optimize(
                program=program,
                train_set=train_set,
                val_set=val_set,
                metric=optimization_metric,
                model_config=self._model_config_to_dict()
            )
            optimizer_trace = getattr(optimizer_adapter, "trace", None)
            if optimizer_trace:
                self.logger.log_optimizer_trace(optimizer_adapter.name, optimizer_trace)
            
            end_time = time.time()
            self.logger.log_optimization_complete(optimizer_adapter.name, end_time - start_time)
            
            # Show prompt changes
            self._log_prompt_comparison(program, optimized_program)
            
            # Show GEPA candidates if available
            if (self.config.optimizer.name == "gepa" and 
                hasattr(optimized_program, "detailed_results")):
                self._log_gepa_candidates(optimized_program)
            
            # Evaluate optimized program
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("OPTIMIZED EVALUATION")
            self.logger.logger.info("=" * 60)
            
            optimized_results = self._evaluate_program(
                optimized_program,
                final_eval_set,
                metric,
                f"Optimized ({final_eval_split})",
            )
            self.logger.log_optimized_results(optimized_results, len(final_eval_set))
            
            # Log comparison
            baseline_score = sum(score for _, _, score in baseline_results.results)
            optimized_score = sum(score for _, _, score in optimized_results.results)
            self.logger.log_comparison_summary(baseline_score, optimized_score, len(final_eval_set))
            
            # Save optimized program if configured
            if self.config.logging.save_models and optimized_program is not None:
                model_path = self._save_optimized_program(optimized_program)
                self.logger.log_program_saved(model_path)
        
        # Save results
        results_path = self.logger.save_results()
        
        return {
            "baseline_results": baseline_results,
            "optimized_results": optimized_results,
            "optimized_program": optimized_program,
            "results_path": results_path
        }
    
    def _setup_model(self) -> None:
        """Setup the language model."""
        lm = dspy.LM(
            model=self.config.model.name,
            api_base=self.config.model.api_base,
            cache=self.config.model.cache,
            **self.config.model.params,
        )
        dspy.configure(lm=lm)
        
        self.logger.log_model_setup(self.config.model.name, self.config.model.cache)
    
    def _evaluate_program(
        self,
        program: Module,
        dataset: list,
        metric: Any,
        name: str
    ) -> Any:
        """Evaluate a program on a dataset."""
        self.logger.logger.info(f"Setting up evaluator for {len(dataset)} examples...")
        self.logger.logger.info(f"Using {self.config.optimizer.num_threads} thread(s)")

        evaluator = dspy.Evaluate(
            devset=dataset,
            metric=metric,
            num_threads=self.config.optimizer.num_threads,
            display_table=False,
            display_progress=True,
        )

        self.logger.logger.info(f"Starting evaluation...")
        results = evaluator(program)
        self.logger.logger.info(f"Evaluation complete")
        
        # Log detailed results
        scores = [score for _, _, score in results.results]
        total_score = sum(scores)
        avg_score = total_score / len(dataset) if dataset else 0
        
        self.logger.logger.info(f"\n{'=' * 50}")
        self.logger.logger.info(f"{name} RESULTS")
        self.logger.logger.info(f"{'=' * 50}")
        self.logger.logger.info(f"Total Score: {total_score:.2f} / {len(dataset)} ({100 * avg_score:.1f}%)")

        for i, (example, prediction, score) in enumerate(results.results):
            # Handle both "question" (HotPotQA) and "problem" (AIME) fields
            question_text = getattr(example, "question", None) or getattr(example, "problem", "")
            question_preview = question_text[:80] if question_text else ""
            pred_answer = getattr(prediction, "answer", str(prediction))
            self.logger.logger.info(f"\n[{i + 1}] Q: {question_preview}...")
            self.logger.logger.info(f"    Gold: {example.answer}")
            self.logger.logger.info(f"    Pred: {pred_answer}")
            self.logger.logger.info(f"    Score: {score:.3f}")
        
        return results
    
    def _log_prompt_comparison(self, baseline_program: Module, optimized_program: Module) -> None:
        """Log comparison of baseline vs optimized prompts."""
        self.logger.logger.info("\n" + "-" * 60)
        self.logger.logger.info("PROMPT COMPARISON")
        self.logger.logger.info("-" * 60)

        baseline_predictors = dict(baseline_program.named_predictors())
        optimized_predictors = dict(optimized_program.named_predictors())

        for pred_name, baseline_pred in baseline_predictors.items():
            optimized_pred = optimized_predictors[pred_name]
            baseline_instr = baseline_pred.signature.instructions
            optimized_instr = optimized_pred.signature.instructions

            self.logger.logger.info(f"\n[PREDICTOR: {pred_name}]")
            self.logger.logger.info(f"  Baseline:")
            self.logger.logger.info(f"    {baseline_instr}")
            self.logger.logger.info(f"  Optimized:")
            self.logger.logger.info(f"    {optimized_instr}")

            status = "[CHANGED]" if baseline_instr != optimized_instr else "[UNCHANGED]"
            self.logger.logger.info(f"  {status}")

        self.logger.logger.info("-" * 60)
    
    def _log_gepa_candidates(self, optimized_program: Module) -> None:
        """Log all candidate instructions proposed by GEPA during optimization."""
        if not hasattr(optimized_program, "detailed_results"):
            self.logger.logger.info("\nNo detailed_results available. Run GEPA with track_stats=True.")
            return

        dr = optimized_program.detailed_results
        
        self.logger.logger.info("\n" + "=" * 60)
        self.logger.logger.info("GEPA CANDIDATE INSPECTION")
        self.logger.logger.info("=" * 60)
        self.logger.logger.info(f"Total candidates explored: {len(dr.candidates)}")
        self.logger.logger.info(f"Best candidate index: {dr.best_idx}")
        self.logger.logger.info(f"Best validation score: {dr.val_aggregate_scores[dr.best_idx]:.4f}")
        
        # Print all candidates with their scores
        self.logger.logger.info("\n" + "-" * 60)
        self.logger.logger.info("ALL PROPOSED CANDIDATES (sorted by score)")
        self.logger.logger.info("-" * 60)
        
        # Sort candidates by score (descending)
        scored_candidates = list(enumerate(zip(dr.candidates, dr.val_aggregate_scores)))
        scored_candidates.sort(key=lambda x: x[1][1], reverse=True)
        
        for rank, (idx, (candidate, score)) in enumerate(scored_candidates, 1):
            best_marker = " ★ BEST" if idx == dr.best_idx else ""
            self.logger.logger.info(f"\n[Rank {rank}] Candidate {idx} — Score: {score:.4f}{best_marker}")
            self.logger.logger.info("-" * 40)
            
            # candidate is a Module, extract predictor instructions
            for pred_name, pred in candidate.named_predictors():
                instruction = pred.signature.instructions
                # Truncate long instructions for display
                if len(instruction) > 500:
                    display_instr = instruction[:500] + "... [truncated]"
                else:
                    display_instr = instruction
                self.logger.logger.info(f"  [{pred_name}]:")
                # Indent multiline instructions
                for line in display_instr.split("\n"):
                    self.logger.logger.info(f"    {line}")
        
        # Print score distribution
        self.logger.logger.info("\n" + "-" * 60)
        self.logger.logger.info("SCORE DISTRIBUTION")
        self.logger.logger.info("-" * 60)
        scores = dr.val_aggregate_scores
        self.logger.logger.info(f"  Min:    {min(scores):.4f}")
        self.logger.logger.info(f"  Max:    {max(scores):.4f}")
        self.logger.logger.info(f"  Mean:   {sum(scores) / len(scores):.4f}")
        
        # Show parent lineage for best candidate
        if dr.best_idx > 0:
            self.logger.logger.info(f"\n  Best candidate lineage: {dr.parents[dr.best_idx]}")
        
        self.logger.logger.info("-" * 60)
    
    def _save_optimized_program(self, program: Module) -> str:
        """Save optimized program to disk.
        
        Saves both the full program (with cloudpickle) and a JSON file 
        with the optimized prompts for easy inspection.
        """
        model_dir = Path(self.config.logging.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.logger.results["start_time"].replace(":", "-").replace(".", "-")
        model_path = model_dir / f"{self.config.name}_{timestamp}.pkl"
        prompts_path = model_dir / f"{self.config.name}_{timestamp}_prompts.json"
        
        # Always save the optimized prompts as JSON (most important!)
        try:
            prompts_data = {
                "experiment": self.config.name,
                "timestamp": self.logger.results["start_time"],
                "predictors": {}
            }
            
            for name, pred in program.named_predictors():
                try:
                    prompts_data["predictors"][name] = {
                        "signature": str(pred.signature),
                        "instructions": pred.signature.instructions,
                        "input_fields": list(pred.signature.input_fields.keys()),
                        "output_fields": list(pred.signature.output_fields.keys())
                    }
                except Exception as e:
                    prompts_data["predictors"][name] = {"error": str(e)}
            
            with open(prompts_path, 'w') as f:
                json.dump(prompts_data, f, indent=2)
            
            self.logger.logger.info(f"Optimized prompts saved to: {prompts_path}")
        except Exception as e:
            self.logger.logger.warning(f"Could not save prompts JSON: {e}")
        
        # Try to save the full program with cloudpickle (handles dynamic classes better)
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(program, f)
            self.logger.logger.info(f"Full program saved with cloudpickle to: {model_path}")
        except Exception as e:
            # If cloudpickle fails, save program state dict as fallback
            self.logger.logger.info(f"Saving program state dict (full pickle failed: {e})")
            try:
                program_state = {
                    "predictors": {},
                    "detailed_results": getattr(program, "detailed_results", None)
                }
                
                for name, pred in program.named_predictors():
                    try:
                        program_state["predictors"][name] = {
                            "signature": str(pred.signature),
                            "instructions": pred.signature.instructions
                        }
                    except:
                        program_state["predictors"][name] = {"error": "Could not serialize"}
                
                with open(model_path, 'wb') as f:
                    pickle.dump(program_state, f)
                    
            except Exception as e2:
                self.logger.logger.warning(f"Could not save program state: {e2}")
                return ""
        
        return str(model_path)
    
    def _config_to_dict(self, config: ExperimentConfig) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "name": config.name,
            "description": config.description,
            "dataset": {
                "name": config.dataset.name,
                "train_size": config.dataset.train_size,
                "dev_size": config.dataset.dev_size,
                "test_size": config.dataset.test_size,
                "params": config.dataset.params
            },
            "model": {
                "name": config.model.name,
                "api_base": config.model.api_base,
                "cache": config.model.cache,
                "params": config.model.params
            },
            "optimizer": {
                "name": config.optimizer.name,
                "auto": config.optimizer.auto,
                "num_threads": config.optimizer.num_threads,
                "params": config.optimizer.params
            },
            "program": {
                "name": config.program.name,
                "params": config.program.params
            }
        }
    
    def _dataset_config_to_dict(self) -> dict[str, Any]:
        """Convert dataset config to dictionary."""
        config_dict = {
            "train_size": self.config.dataset.train_size,
            "dev_size": self.config.dataset.dev_size,
            "test_size": self.config.dataset.test_size,
            "train_seed": self.config.dataset.train_seed,
            "eval_seed": self.config.dataset.eval_seed,
            "keep_details": self.config.dataset.keep_details,
        }
        config_dict.update(self.config.dataset.params)
        return config_dict
    
    def _model_config_to_dict(self) -> dict[str, Any]:
        """Convert model config to dictionary."""
        config_dict = {
            "name": self.config.model.name,
            "api_base": self.config.model.api_base,
            "cache": self.config.model.cache,
        }
        config_dict.update(self.config.model.params)
        return config_dict
    
    def _optimizer_config_to_dict(self) -> dict[str, Any]:
        """Convert optimizer config to dictionary."""
        config_dict = {
            "auto": self.config.optimizer.auto,
            "num_threads": self.config.optimizer.num_threads,
            "params": self.config.optimizer.params or {}
        }
        return config_dict
