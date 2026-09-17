from __future__ import annotations

import re

from .rag import retrieve_use_cases
from .schemas import ProjectSpec, TaskType

CLASSIFICATION_WORDS = {"classify", "classification", "churn", "fraud", "default", "delay", "approve", "yes", "no", "likely", "risk"}
REGRESSION_WORDS = {"predict price", "forecast", "revenue", "sales", "amount", "cost", "demand", "temperature", "duration"}


def infer_spec(problem: str) -> ProjectSpec:
    text = problem.lower()
    classification_hits = sum(word in text for word in CLASSIFICATION_WORDS)
    regression_hits = sum(word in text for word in REGRESSION_WORDS)
    task_type = None
    if classification_hits > regression_hits and classification_hits:
        task_type = TaskType.CLASSIFICATION
    elif regression_hits > classification_hits and regression_hits:
        task_type = TaskType.REGRESSION
    metric = "f1" if task_type == TaskType.CLASSIFICATION else "mae" if task_type == TaskType.REGRESSION else None
    target = _extract_target(problem)
    recommendations = retrieve_use_cases(text, task_type)
    missing = []
    if task_type is None:
        missing.append("Should the outcome be a category (classification) or a numeric value (regression)?")
    if not target:
        missing.append("What is the historical outcome column the model should learn to predict?")
    missing.append("Which fields are available before the prediction needs to be made?")
    return ProjectSpec(
        business_objective=problem.strip(), task_type=task_type, target_column=target,
        prediction_unit=None, primary_metric=metric,
        required_fields=["historical outcome/target", "predictive fields available before the decision"],
        recommendations=recommendations,
        assumptions=["Only tabular historical data is supported in this MVP."],
        missing_information=missing,
        readiness="needs_clarification" if missing else "ready_for_approval",
    )


def _extract_target(problem: str) -> str | None:
    explicit = re.search(
        r"(?:target\s+column|outcome\s+column|target|label)\s*(?:is|as|=|:)\s*[`\"']?([A-Za-z][A-Za-z0-9_]{0,127})",
        problem,
        re.I,
    )
    return explicit.group(1).strip() if explicit else None


def apply_message(spec: dict, message: str) -> dict:
    updated = dict(spec)
    target = _extract_target(message)
    if target:
        updated["target_column"] = target
    text = message.lower()
    if "classification" in text:
        updated["task_type"], updated["primary_metric"] = "classification", "f1"
    elif "regression" in text:
        updated["task_type"], updated["primary_metric"] = "regression", "mae"
    updated["missing_information"] = [item for item in updated.get("missing_information", []) if "outcome column" not in item or not updated.get("target_column")]
    updated["readiness"] = "ready_for_approval" if updated.get("task_type") and updated.get("target_column") else "needs_clarification"
    return updated
