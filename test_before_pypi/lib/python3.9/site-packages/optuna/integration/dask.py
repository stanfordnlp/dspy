from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.dask import DaskStorage
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("dask"))


__all__ = ["DaskStorage"]
