from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ProjectStatus(str, Enum):
    DISCOVERY = "discovery"
    AWAITING_APPROVAL = "awaiting_approval"
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


class RunResponse(BaseModel):
    id: str
    project_id: str
    status: str
    result: dict | None = None
    error: str | None = None
