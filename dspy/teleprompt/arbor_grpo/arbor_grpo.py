from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Union

from dspy.clients.lm import LM
from dspy.clients.utils_finetune import MultiGPUConfig
from dspy.teleprompt.grpo import GRPO

from .arbor_grpo_config import ArborGRPOConfig


class ArborGRPO(GRPO):
    """
    GRPO variant that accepts ArborGRPOConfig and ensures LM.reinforce receives it.
    """

    def __init__(
        self,
        *args,
        config: Optional[Union[ArborGRPOConfig, Dict[LM, ArborGRPOConfig]]] = None,
        gpu_config: MultiGPUConfig = MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gpu_config = gpu_config
        self.grpo_configs: Dict[LM, ArborGRPOConfig] = self._normalize_config_dict(config)

    def _normalize_config_dict(
        self, cfg: Optional[Union[ArborGRPOConfig, Dict[LM, ArborGRPOConfig]]]
    ) -> Dict[LM, ArborGRPOConfig]:
        if cfg is None:
            return defaultdict(lambda: ArborGRPOConfig(num_generations=1))  # type: ignore[return-value]
        if isinstance(cfg, ArborGRPOConfig):
            return defaultdict(lambda: cfg)  # type: ignore[return-value]
        # Validate user dict: LM -> ArborGRPOConfig
        if not all(isinstance(k, LM) for k in cfg.keys()):
            raise ValueError("config dict must have LM keys")
        if not all(isinstance(v, ArborGRPOConfig) for v in cfg.values()):
            raise ValueError("config dict must have ArborGRPOConfig values")
        return dict(cfg)

    def compile(self, student, trainset, teacher=None, valset=None):  # type: ignore[override]
        # Ensure the provider-side sampler matches GRPO’s per-input sample count
        try:
            num_samples = getattr(self, "num_samples_per_input", None)
            if num_samples is None:
                # Older GRPO naming
                num_samples = getattr(self, "num_rollouts_per_grpo_step", 1)
            for pred in student.predictors():
                self.grpo_configs[pred.lm].num_generations = int(num_samples)
        except Exception:
            pass

        # Monkey-patch student LMs so .reinforce(...) always gets our ArborGRPOConfig
        originals = {}
        try:
            for pred in student.predictors():
                lm = pred.lm
                if lm in originals:
                    continue
                originals[lm] = lm.reinforce
                cfg = self.grpo_configs[lm]

                def make_wrapper(lm_ref, cfg_ref, orig_ref):
                    def wrapper(*args, **kwargs):
                        kwargs.pop("train_kwargs", None)  # drop legacy path
                        kwargs.setdefault("config", cfg_ref)
                        kwargs.setdefault("gpu_config", self.gpu_config)
                        return orig_ref(*args, **kwargs)
                    return wrapper

                lm.reinforce = make_wrapper(lm, cfg, originals[lm])  # type: ignore[method-assign]

            return super().compile(student=student, trainset=trainset, teacher=teacher, valset=valset)
        finally:
            # Restore original methods
            for lm, fn in originals.items():
                lm.reinforce = fn  # type: ignore[method-assign]


