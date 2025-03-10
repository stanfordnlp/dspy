from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("pytorch_lightning"))


__all__ = ["PyTorchLightningPruningCallback"]
