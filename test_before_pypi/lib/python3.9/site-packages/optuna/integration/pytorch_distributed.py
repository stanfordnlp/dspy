from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.pytorch_distributed import TorchDistributedTrial
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("pytorch_distributed"))


__all__ = ["TorchDistributedTrial"]
