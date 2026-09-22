from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def predict_records(artifact_path: Path, expected_features: list[str], records: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, float]] | None]:
    frame = pd.DataFrame(records)
    missing_features = [feature for feature in expected_features if feature not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")
    model = joblib.load(artifact_path)
    values = model.predict(frame[expected_features])
    predictions = [_json_value(value) for value in values]
    if not hasattr(model, "predict_proba"):
        return predictions, None
    probabilities = model.predict_proba(frame[expected_features])
    classes = model.named_steps["model"].classes_
    return predictions, [
        {str(_json_value(label)): float(probability) for label, probability in zip(classes, row)}
        for row in probabilities
    ]


def _json_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
