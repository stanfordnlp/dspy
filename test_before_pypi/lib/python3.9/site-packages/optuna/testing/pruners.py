from __future__ import annotations

import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning: bool) -> None:
        self.is_pruning = is_pruning

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        return self.is_pruning
