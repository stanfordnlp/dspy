from optuna._imports import _INTEGRATION_IMPORT_ERROR_TEMPLATE


try:
    from optuna_integration.shap import ShapleyImportanceEvaluator
except ModuleNotFoundError:
    raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("shap"))


__all__ = ["ShapleyImportanceEvaluator"]
