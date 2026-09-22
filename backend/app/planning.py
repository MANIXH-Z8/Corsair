from __future__ import annotations

from typing import Any

from .schemas import TaskType


_CANDIDATES = {
    TaskType.CLASSIFICATION.value: [
        ("logistic_regression", "Strong interpretable baseline for binary and multiclass outcomes."),
        ("random_forest", "Captures nonlinear relationships and mixed feature effects."),
        ("hist_gradient_boosting", "Tests a boosted-tree candidate for structured tabular data."),
    ],
    TaskType.REGRESSION.value: [
        ("ridge", "Regularized linear baseline that is stable with correlated features."),
        ("random_forest", "Captures nonlinear relationships and feature interactions."),
        ("hist_gradient_boosting", "Tests a boosted-tree candidate for structured tabular data."),
    ],
}


def build_run_plan(spec: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    task_type = spec.get("task_type")
    if task_type not in _CANDIDATES:
        raise ValueError("A classification or regression task is required before planning a run")
    metric = spec.get("primary_metric") or ("f1" if task_type == TaskType.CLASSIFICATION.value else "mae")
    candidates = [{"name": name, "reason": reason} for name, reason in _CANDIDATES[task_type]]
    plan = {
        "task_type": task_type,
        "target_column": spec.get("target_column"),
        "primary_metric": metric,
        "metric_direction": "higher_is_better" if metric == "f1" else "lower_is_better",
        "validation": "stratified 5-fold cross-validation" if task_type == TaskType.CLASSIFICATION.value else "5-fold cross-validation",
        "candidates": candidates,
        "reproducibility": {"random_seed": 42, "preprocessing": "imputation, one-hot encoding, and scaling inside each validation fold"},
    }
    if profile:
        plan["dataset"] = {"rows": profile.get("rows"), "columns": profile.get("columns"), "features": [column for column in profile.get("column_names", []) if column != spec.get("target_column")]}
    return plan
