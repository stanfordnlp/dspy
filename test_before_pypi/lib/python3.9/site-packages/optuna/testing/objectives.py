from optuna import TrialPruned
from optuna.trial import Trial


def fail_objective(_: Trial) -> float:
    raise ValueError()


def pruned_objective(trial: Trial) -> float:
    raise TrialPruned()
