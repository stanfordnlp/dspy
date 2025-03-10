import optuna
from optuna.pruners import BasePruner


class NopPruner(BasePruner):
    """Pruner which never prunes trials.

    Example:

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        assert False, "should_prune() should always return False with this pruner."
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
            study.optimize(objective, n_trials=20)
    """

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        return False
