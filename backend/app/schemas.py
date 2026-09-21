from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ProjectStatus(str, Enum):
    DISCOVERY = "discovery"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_DATA = "awaiting_data"
    DATA_READY = "data_ready"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    problem_statement: str = Field(min_length=10, max_length=4000)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ProjectSpec(BaseModel):
    business_objective: str
    task_type: TaskType | None = None
    target_column: str | None = None
    prediction_unit: str | None = None
    primary_metric: str | None = None
    required_fields: list[str] = []
    recommendations: list[dict] = []
    assumptions: list[str] = []
    missing_information: list[str] = []
    readiness: str


class ApprovalRequest(BaseModel):
    task_type: TaskType
    target_column: str = Field(min_length=1, max_length=128)
    primary_metric: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: ProjectStatus
    problem_statement: str
    spec: ProjectSpec


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    profile: dict
    training_ready: bool
    blockers: list[str]
    warnings: list[str] = []


class RunResponse(BaseModel):
    id: str
    project_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    artifact_available: bool = False


class RunPlanResponse(BaseModel):
    task_type: TaskType
    target_column: str
    primary_metric: str
    metric_direction: str
    validation: str
    candidates: list[dict[str, str]]
    reproducibility: dict[str, Any]
    dataset: dict[str, Any] | None = None


class WorkflowStage(BaseModel):
    id: str
    label: str
    state: str
    description: str


class WorkflowEvent(BaseModel):
    id: int
    step: str
    detail: str
    created_at: str


class WorkflowResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    current_stage: str
    next_action: str
    stages: list[WorkflowStage]
    events: list[WorkflowEvent]
