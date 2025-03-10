from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.skorch import SkorchPruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("skorch"))


__all__ = ["SkorchPruningCallback"]
