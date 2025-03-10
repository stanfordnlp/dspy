from optuna import Study
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.trial import create_trial


def prepare_study_with_trials(
    n_objectives: int = 1,
    direction: str = "minimize",
    value_for_first_trial: float = 0.0,
) -> Study:
    """Return a dummy study object for tests.

    This function is added to reduce the code to set up dummy study object in each test case.
    However, you can only use this function for unit tests that are loosely coupled with the
    dummy study object. Unit tests that are tightly coupled with the study become difficult to
    read because of
    `Mystery Guest <http://xunitpatterns.com/Obscure%20Test.html#Mystery%20Guest>`__ and/or
    `Eager Test <http://xunitpatterns.com/Obscure%20Test.html#Eager%20Test>`__ anti-patterns.

    Args:
        n_objectives: Number of objective values.
        direction: Study's optimization direction.
        value_for_first_trial: Objective value in first trial. This value will be broadcasted
            to all objectives in multi-objective optimization.

    Returns:
        :class:`~optuna.study.Study`

    """

    study = create_study(directions=[direction] * n_objectives)
    study.add_trial(
        create_trial(
            values=[value_for_first_trial] * n_objectives,
            params={"param_a": 1.0, "param_b": 2.0, "param_c": 3.0, "param_d": 4.0},
            distributions={
                "param_a": FloatDistribution(0.0, 3.0),
                "param_b": FloatDistribution(0.0, 3.0),
                "param_c": FloatDistribution(2.0, 5.0),
                "param_d": FloatDistribution(2.0, 5.0),
            },
        )
    )
    study.add_trial(
        create_trial(
            values=[2.0] * n_objectives,
            params={"param_b": 0.0, "param_d": 4.0},
            distributions={
                "param_b": FloatDistribution(0.0, 3.0),
                "param_d": FloatDistribution(2.0, 5.0),
            },
        )
    )
    study.add_trial(
        create_trial(
            values=[1.0] * n_objectives,
            params={"param_a": 2.5, "param_b": 1.0, "param_c": 4.5, "param_d": 2.0},
            distributions={
                "param_a": FloatDistribution(0.0, 3.0),
                "param_b": FloatDistribution(0.0, 3.0),
                "param_c": FloatDistribution(2.0, 5.0),
                "param_d": FloatDistribution(2.0, 5.0),
            },
        )
    )
    return study
