from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def profile_dataset(path: Path, target: str | None, task_type: str | None) -> tuple[dict, list[str]]:
    frame = read_table(path)
    blockers: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        blockers.append("The file contains no rows.")
    if not target:
        blockers.append("Approve a project specification with a target column before training.")
    elif target not in frame.columns:
        blockers.append(f"Target column '{target}' is not present in the imported file.")
    elif task_type not in {"classification", "regression"}:
        blockers.append("Approve whether this is a classification or regression project before training.")
    else:
        labeled_target = frame[target].dropna()
        if labeled_target.nunique() < 2:
            blockers.append("The target must contain at least two distinct non-empty values.")
        if task_type == "classification" and not labeled_target.empty and labeled_target.value_counts().min() < 2:
            blockers.append("Each target class needs at least two labeled rows for cross-validation.")
        if task_type == "regression" and not pd.api.types.is_numeric_dtype(labeled_target):
            blockers.append("A regression target must be numeric. Use classification for category labels.")
        missing_target = int(frame[target].isna().sum())
        if missing_target:
            warnings.append(f"{missing_target} row(s) with a missing target will be excluded from training.")
    if len(frame) < 30:
        blockers.append("At least 30 rows are required for this MVP to evaluate models safely.")
    feature_columns = [column for column in frame.columns if column != target]
    empty_features = [column for column in feature_columns if frame[column].dropna().empty]
    constant_features = [column for column in feature_columns if column not in empty_features and frame[column].dropna().nunique() <= 1]
    usable_features = [column for column in feature_columns if column not in empty_features and column not in constant_features]
    if not feature_columns:
        blockers.append("Add at least one feature column in addition to the target column.")
    elif not usable_features:
        blockers.append("Add at least one non-empty feature column with more than one value.")
    if empty_features:
        warnings.append(f"{len(empty_features)} fully empty feature column(s) will not provide predictive signal: {', '.join(empty_features[:5])}.")
    if constant_features:
        warnings.append(f"{len(constant_features)} constant feature column(s) will not provide predictive signal: {', '.join(constant_features[:5])}.")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) were detected; review whether repeated records are expected.")
    profile = {
        "rows": len(frame), "columns": len(frame.columns), "column_names": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": frame.isna().sum().to_dict(), "duplicate_rows": duplicate_rows,
        "target": target, "task_type": task_type,
        "target_distribution": frame[target].value_counts(dropna=False).head(20).to_dict() if target in frame else {},
        "quality": {"feature_columns": feature_columns, "usable_feature_columns": usable_features, "empty_feature_columns": empty_features, "constant_feature_columns": constant_features},
        "warnings": warnings,
    }
    return profile, blockers


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)


def train(path: Path, target: str, task_type: str, metric: str, artifact_path: Path) -> dict:
    frame = read_table(path).dropna(subset=[target]).copy()
    x = frame.drop(columns=[target])
    y = frame[target]
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
    if task_type == "classification":
        candidates = {"logistic_regression": LogisticRegression(max_iter=1000), "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1), "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42)}
        cv = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=42)
        scoring, direction = "f1_weighted", "max"
    else:
        candidates = {"ridge": Ridge(), "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1), "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42)}
        cv = KFold(n_splits=min(5, len(frame)), shuffle=True, random_state=42)
        scoring, direction = "neg_mean_absolute_error", "max"
    results = []
    best_name, best_pipeline, best_score = "", None, float("-inf")
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        scores = cross_val_score(pipeline, x, y, scoring=scoring, cv=cv, error_score="raise")
        score = float(scores.mean()) if task_type == "classification" else float(-scores.mean())
        results.append({"model": name, "metric": metric, "score": score, "fold_std": float(scores.std())})
        comparable = score if task_type == "classification" else -score
        if comparable > best_score:
            best_name, best_pipeline, best_score = name, pipeline, comparable
    best_pipeline.fit(x, y)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifact_path)
    results.sort(key=lambda item: item["score"], reverse=task_type == "classification")
    return {"primary_metric": metric, "leaderboard": results, "best_model": best_name, "artifact": str(artifact_path.name), "rows_used": len(frame), "features_used": x.columns.tolist(), "random_seed": 42}
