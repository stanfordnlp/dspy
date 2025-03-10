from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.tensorflow import TensorFlowPruningHook
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("tensorflow"))


__all__ = ["TensorFlowPruningHook"]
