"""
Logging and results tracking for prompt optimization experiments.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy


class ExperimentLogger:
    """Logger for tracking experiment progress and results."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "results/logs",
        level: str = "INFO",
        log_to_file: bool = True,
        include_timestamps: bool = True
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.include_timestamps = include_timestamps
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = "%(levelname)s - %(message)s"
        if include_timestamps:
            console_format = "%(asctime)s - " + console_format
        console_handler.setFormatter(logging.Formatter(console_format))
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_format = "%(asctime)s - %(levelname)s - %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format))
            self.logger.addHandler(file_handler)
            
        # Results tracking
        self.results: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "baseline_results": {},
            "optimization_results": {},
            "final_results": {},
            "timeline": []
        }
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.results["config"] = config
        self.logger.info(f"Experiment config: {json.dumps(config, indent=2)}")
        self._add_timeline_event("config_loaded", config)
    
    def log_dataset_loaded(self, dataset_name: str, train_size: int, val_size: int, test_size: int) -> None:
        """Log dataset loading information."""
        info = {
            "dataset": dataset_name,
            "train_size": train_size,
            "val_size": val_size, 
            "test_size": test_size
        }
        self.logger.info(f"Dataset loaded: {dataset_name} (train={train_size}, val={val_size}, test={test_size})")
        self._add_timeline_event("dataset_loaded", info)
    
    def log_model_setup(self, model_name: str, cache_enabled: bool) -> None:
        """Log model setup information."""
        info = {"model": model_name, "cache": cache_enabled}
        self.logger.info(f"Model configured: {model_name} (cache: {'enabled' if cache_enabled else 'disabled'})")
        self._add_timeline_event("model_setup", info)
    
    def log_baseline_results(self, results: Any, dataset_size: int) -> None:
        """Log baseline evaluation results."""
        scores = [score for _, _, score in results.results]
        total_score = sum(scores)
        avg_score = total_score / dataset_size if dataset_size > 0 else 0
        
        baseline_info = {
            "total_score": total_score,
            "avg_score": avg_score,
            "dataset_size": dataset_size,
            "individual_scores": scores
        }
        
        self.results["baseline_results"] = baseline_info
        self.logger.info(f"Baseline results: {total_score:.2f}/{dataset_size} ({100*avg_score:.1f}%)")
        self._add_timeline_event("baseline_evaluated", baseline_info)
    
    def log_optimization_start(self, optimizer_name: str) -> None:
        """Log start of optimization."""
        self.logger.info(f"Starting optimization with {optimizer_name}")
        self._add_timeline_event("optimization_started", {"optimizer": optimizer_name})
    
    def log_optimization_complete(self, optimizer_name: str, duration_seconds: float) -> None:
        """Log completion of optimization."""
        info = {"optimizer": optimizer_name, "duration": duration_seconds}
        self.logger.info(f"Optimization complete: {optimizer_name} ({duration_seconds:.1f}s)")
        self._add_timeline_event("optimization_completed", info)
    
    def log_optimized_results(self, results: Any, dataset_size: int) -> None:
        """Log optimized model evaluation results."""
        scores = [score for _, _, score in results.results]
        total_score = sum(scores)
        avg_score = total_score / dataset_size if dataset_size > 0 else 0
        
        optimized_info = {
            "total_score": total_score,
            "avg_score": avg_score,
            "dataset_size": dataset_size,
            "individual_scores": scores
        }
        
        self.results["optimization_results"] = optimized_info
        self.logger.info(f"Optimized results: {total_score:.2f}/{dataset_size} ({100*avg_score:.1f}%)")
        self._add_timeline_event("optimized_evaluated", optimized_info)
    
    def log_comparison_summary(self, baseline_score: float, optimized_score: float, dataset_size: int) -> None:
        """Log comparison between baseline and optimized results."""
        improvement = optimized_score - baseline_score
        pct_improvement = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        comparison = {
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
            "dataset_size": dataset_size
        }
        
        self.results["final_results"] = comparison
        self.logger.info(f"Improvement: {improvement:+.2f} ({pct_improvement:+.1f}% relative)")
        self._add_timeline_event("comparison_summary", comparison)
    
    def log_program_saved(self, save_path: str) -> None:
        """Log that optimized program was saved."""
        self.logger.info(f"Optimized program saved to: {save_path}")
        self._add_timeline_event("program_saved", {"path": save_path})
    
    def save_results(self) -> str:
        """Save experiment results to JSON file."""
        self.results["end_time"] = datetime.now().isoformat()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.log_dir / f"{self.experiment_name}_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
        return str(results_file)
    
    def _add_timeline_event(self, event_type: str, data: Any) -> None:
        """Add event to experiment timeline."""
        self.results["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data
        })


def setup_logging(experiment_name: str, config: Dict[str, Any]) -> ExperimentLogger:
    """Setup experiment logging with given configuration.
    
    Args:
        experiment_name: Name of the experiment.
        config: Logging configuration dictionary.
        
    Returns:
        Configured experiment logger.
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=config.get("log_dir", "results/logs"),
        level=config.get("level", "INFO"),
        log_to_file=config.get("log_to_file", True),
        include_timestamps=config.get("include_timestamps", True)
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)