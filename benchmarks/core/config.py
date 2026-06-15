"""
Configuration management for prompt optimization experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    train_size: int = 50
    dev_size: int = 100
    test_size: int = 0
    train_seed: int = 1
    eval_seed: int = 2023
    keep_details: bool = True
    # Additional dataset-specific parameters
    params: dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass  
class ModelConfig:
    """Model configuration."""
    name: str = "ollama_chat/qwen3:4b-instruct"
    api_base: str = "http://localhost:11434"
    cache: bool = True
    # Additional model-specific parameters
    params: dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str
    auto: str = "light"
    num_threads: int = 1
    # Optimizer-specific parameters
    params: dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ProgramConfig:
    """Program configuration."""
    name: str = "naive"
    # Program-specific parameters
    params: dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "results/logs"
    include_timestamps: bool = True
    save_models: bool = True
    model_dir: str = "results/models"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    dataset: DatasetConfig = None
    model: ModelConfig = None
    optimizer: OptimizerConfig = None
    program: ProgramConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetConfig(name="hotpotqa")
        if self.model is None:
            self.model = ModelConfig()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig(name="baseline")
        if self.program is None:
            self.program = ProgramConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.
    
    Supports modular configs via references:
    - dataset_ref: references configs/datasets/<name>.yaml
    - model_ref: references configs/models/<name>.yaml
    - optimizer_ref: references configs/optimizers/<name>.yaml
    - program_ref: references configs/programs/<name>.yaml
    - logging_ref: references configs/base/<name>.yaml
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Parsed experiment configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config format is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a YAML dictionary")
    
    # Resolve references to modular configs
    config_dict = _resolve_references(config_dict, config_path)
    
    return _parse_config_dict(config_dict)


def save_config(config: ExperimentConfig, config_path: str | Path) -> None:
    """Save experiment configuration to YAML file.
    
    Args:
        config: Experiment configuration to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = _config_to_dict(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _resolve_references(config_dict: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Resolve modular config references.
    
    Args:
        config_dict: The experiment config dictionary.
        config_path: Path to the main config file (used to find config root).
        
    Returns:
        Config dictionary with references resolved and merged.
    """
    # Find config root (go up from experiments/ to configs/)
    config_root = config_path.parent.parent
    
    # Map of reference keys to their subdirectories
    ref_map = {
        'dataset_ref': 'datasets',
        'model_ref': 'models',
        'optimizer_ref': 'optimizers',
        'program_ref': 'programs',
        'logging_ref': 'base'
    }
    
    resolved = config_dict.copy()
    
    for ref_key, subdir in ref_map.items():
        if ref_key in config_dict:
            ref_name = config_dict[ref_key]
            ref_path = config_root / subdir / f"{ref_name}.yaml"
            
            if not ref_path.exists():
                raise FileNotFoundError(f"Referenced config not found: {ref_path}")
            
            with open(ref_path) as f:
                ref_config = yaml.safe_load(f)
            
            # Determine the target key (remove _ref suffix)
            target_key = ref_key.replace('_ref', '')
            
            # Merge: referenced config as base, then apply overrides
            override_key = f"{target_key}_overrides"
            if override_key in config_dict:
                # Deep merge overrides
                merged = _deep_merge(ref_config, config_dict[override_key])
                resolved[target_key] = merged
            else:
                resolved[target_key] = ref_config
            
            # Remove the reference and override keys
            resolved.pop(ref_key, None)
            resolved.pop(override_key, None)
    
    return resolved


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary.
        override: Override dictionary (takes precedence).
        
    Returns:
        Merged dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _parse_config_dict(config_dict: dict[str, Any]) -> ExperimentConfig:
    """Parse configuration dictionary into ExperimentConfig."""
    
    # Parse dataset config
    dataset_dict = config_dict.get("dataset", {})
    dataset = DatasetConfig(
        name=dataset_dict["name"],
        train_size=dataset_dict.get("train_size", 50),
        dev_size=dataset_dict.get("dev_size", 100), 
        test_size=dataset_dict.get("test_size", 0),
        train_seed=dataset_dict.get("train_seed", 1),
        eval_seed=dataset_dict.get("eval_seed", 2023),
        keep_details=dataset_dict.get("keep_details", True),
        params=dataset_dict.get("params", {})
    )
    
    # Parse model config
    model_dict = config_dict.get("model", {})
    model = ModelConfig(
        name=model_dict.get("name", "ollama_chat/qwen3:4b-instruct"),
        api_base=model_dict.get("api_base", "http://localhost:11434"),
        cache=model_dict.get("cache", True),
        params=model_dict.get("params", {})
    )
    
    # Parse optimizer config
    optimizer_dict = config_dict.get("optimizer", {})
    optimizer = OptimizerConfig(
        name=optimizer_dict["name"],
        auto=optimizer_dict.get("auto", "light"),
        num_threads=optimizer_dict.get("num_threads", 1),
        params=optimizer_dict.get("params", {})
    )
    
    # Parse program config
    program_dict = config_dict.get("program", {})
    program = ProgramConfig(
        name=program_dict.get("name", "naive"),
        params=program_dict.get("params", {})
    )
    
    # Parse logging config
    logging_dict = config_dict.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_dict.get("level", "INFO"),
        log_to_file=logging_dict.get("log_to_file", True),
        log_dir=logging_dict.get("log_dir", "results/logs"),
        include_timestamps=logging_dict.get("include_timestamps", True),
        save_models=logging_dict.get("save_models", True),
        model_dir=logging_dict.get("model_dir", "results/models")
    )
    
    return ExperimentConfig(
        name=config_dict["name"],
        description=config_dict.get("description", ""),
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        program=program,
        logging=logging_config
    )


def _config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert ExperimentConfig to dictionary."""
    return {
        "name": config.name,
        "description": config.description,
        "dataset": {
            "name": config.dataset.name,
            "train_size": config.dataset.train_size,
            "dev_size": config.dataset.dev_size,
            "test_size": config.dataset.test_size,
            "train_seed": config.dataset.train_seed,
            "eval_seed": config.dataset.eval_seed,
            "keep_details": config.dataset.keep_details,
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
        },
        "logging": {
            "level": config.logging.level,
            "log_to_file": config.logging.log_to_file,
            "log_dir": config.logging.log_dir,
            "include_timestamps": config.logging.include_timestamps,
            "save_models": config.logging.save_models,
            "model_dir": config.logging.model_dir
        }
    }