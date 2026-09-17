from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import TaskType

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "use_cases.json"


def retrieve_use_cases(problem_text: str, task_type: TaskType | str | None, limit: int = 3) -> list[dict]:
    """Retrieve reviewed local use cases for the user's tabular ML problem."""
    cases = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    tokens = set(re.findall(r"[a-z]+", problem_text.lower()))
    task_value = task_type.value if isinstance(task_type, TaskType) else task_type
    ranked: list[tuple[int, dict]] = []

    for case in cases:
        if task_value and case["task_type"] != task_value:
            continue
        overlap = len(tokens.intersection(case["keywords"]))
        if overlap:
            ranked.append((overlap, {key: case[key] for key in ("id", "task_type", "recommended_features", "model_notes")}))

    return [case for _, case in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]
