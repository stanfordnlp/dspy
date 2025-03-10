from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.fastaiv2 import FastAIPruningCallback
    from optuna_integration.fastaiv2 import FastAIV2PruningCallback
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("fastaiv2"))


__all__ = ["FastAIV2PruningCallback", "FastAIPruningCallback"]
