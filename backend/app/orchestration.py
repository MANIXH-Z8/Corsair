from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from typing_extensions import TypedDict

from .discovery import apply_message, infer_spec
from .schemas import ProjectSpec, ProjectStatus, WorkflowStage

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - local fallback until dependency install
    END = "__end__"
    START = "__start__"
    StateGraph = None


class DiscoveryState(TypedDict, total=False):
    operation: Literal["create", "refine"]
    problem_statement: str
    message: str
    current_spec: dict[str, Any]
    spec: dict[str, Any]
    events: list[str]


class LifecycleState(TypedDict, total=False):
    status: str
    spec: dict[str, Any]
    has_dataset: bool
    has_completed_run: bool
    current_stage: str
    next_action: str
    stages: list[dict[str, str]]


class _SequentialDiscoveryGraph:
    def invoke(self, state: DiscoveryState) -> DiscoveryState:
        state = _route_discovery(state)
        state = _validate_spec(state)
        return state


def run_initial_discovery(problem_statement: str) -> ProjectSpec:
    state = _compiled_discovery_graph().invoke({"operation": "create", "problem_statement": problem_statement})
    return ProjectSpec(**state["spec"])


def run_followup_discovery(current_spec: dict[str, Any], message: str) -> ProjectSpec:
    state = _compiled_discovery_graph().invoke({"operation": "refine", "current_spec": current_spec, "message": message})
    return ProjectSpec(**state["spec"])


def workflow_summary(
    status: ProjectStatus | str, spec: dict[str, Any], has_dataset: bool, has_completed_run: bool
) -> tuple[str, str, list[WorkflowStage]]:
    state = _compiled_lifecycle_graph().invoke(
        {"status": status.value if isinstance(status, ProjectStatus) else status, "spec": spec, "has_dataset": has_dataset, "has_completed_run": has_completed_run}
    )
    return state["current_stage"], state["next_action"], [WorkflowStage(**item) for item in state["stages"]]


@lru_cache(maxsize=1)
def _compiled_discovery_graph():
    if StateGraph is None:
        return _SequentialDiscoveryGraph()

    workflow = StateGraph(DiscoveryState)
    workflow.add_node("route_discovery", _route_discovery)
    workflow.add_node("validate_spec", _validate_spec)
    workflow.add_edge(START, "route_discovery")
    workflow.add_edge("route_discovery", "validate_spec")
    workflow.add_edge("validate_spec", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def _compiled_lifecycle_graph():
    if StateGraph is None:
        return _SequentialLifecycleGraph()

    workflow = StateGraph(LifecycleState)
    workflow.add_node("assess_spec", _assess_spec)
    workflow.add_node("assess_data", _assess_data)
    workflow.add_node("assess_run", _assess_run)
    workflow.add_edge(START, "assess_spec")
    workflow.add_edge("assess_spec", "assess_data")
    workflow.add_edge("assess_data", "assess_run")
    workflow.add_edge("assess_run", END)
    return workflow.compile()


def _route_discovery(state: DiscoveryState) -> DiscoveryState:
    if state["operation"] == "create":
        spec = infer_spec(state["problem_statement"]).model_dump()
        return {**state, "spec": spec, "events": ["problem_statement_inferred"]}

    spec = apply_message(state["current_spec"], state["message"])
    return {**state, "spec": spec, "events": ["project_spec_refined"]}


def _validate_spec(state: DiscoveryState) -> DiscoveryState:
    spec = ProjectSpec(**state["spec"]).model_dump()
    missing = list(spec.get("missing_information") or [])

    if spec.get("task_type") and spec.get("target_column"):
        missing = [item for item in missing if "category" not in item.lower() and "outcome column" not in item.lower()]
        spec["readiness"] = "ready_for_approval"
    else:
        spec["readiness"] = "needs_clarification"

    spec["missing_information"] = missing
    events = [*state.get("events", []), "project_spec_validated"]
    return {**state, "spec": spec, "events": events}


class _SequentialLifecycleGraph:
    def invoke(self, state: LifecycleState) -> LifecycleState:
        return _assess_run(_assess_data(_assess_spec(state)))


def _assess_spec(state: LifecycleState) -> LifecycleState:
    status = state["status"]
    if status == ProjectStatus.DISCOVERY.value:
        current, action = "discovery", "Answer the remaining discovery questions."
    elif status == ProjectStatus.AWAITING_APPROVAL.value:
        current, action = "approval", "Review and approve the task type, target column, and metric."
    else:
        current, action = "data_upload", "Upload historical CSV or XLSX data, or download the generated template."
    stages = [
        {"id": "discovery", "label": "Describe the problem", "state": "current" if current == "discovery" else "complete", "description": "Turn the business problem into a prediction brief."},
        {"id": "approval", "label": "Approve the brief", "state": "current" if current == "approval" else ("complete" if status not in {ProjectStatus.DISCOVERY.value, ProjectStatus.AWAITING_APPROVAL.value} else "pending"), "description": "Confirm task type, outcome column, and evaluation metric."},
    ]
    return {**state, "current_stage": current, "next_action": action, "stages": stages}


def _assess_data(state: LifecycleState) -> LifecycleState:
    status = state["status"]
    state_name = "complete" if state["has_dataset"] and status in {ProjectStatus.DATA_READY.value, ProjectStatus.RUNNING.value, ProjectStatus.COMPLETED.value} else "current" if status in {ProjectStatus.AWAITING_DATA.value, ProjectStatus.BLOCKED.value, ProjectStatus.DATA_READY.value} else "pending"
    stages = [*state["stages"], {"id": "data_upload", "label": "Check historical data", "state": state_name, "description": "Upload data and resolve any profile blockers."}]
    if status == ProjectStatus.BLOCKED.value:
        return {**state, "stages": stages, "current_stage": "data_upload", "next_action": "Resolve the data blockers and upload a corrected file."}
    return {**state, "stages": stages}


def _assess_run(state: LifecycleState) -> LifecycleState:
    status = state["status"]
    run_state = "complete" if state["has_completed_run"] else "current" if status in {ProjectStatus.DATA_READY.value, ProjectStatus.RUNNING.value, ProjectStatus.FAILED.value} else "pending"
    stages = [*state["stages"], {"id": "training", "label": "Train and compare models", "state": run_state, "description": "Run validated model candidates and compare them using the chosen metric."}]
    if status == ProjectStatus.DATA_READY.value:
        return {**state, "stages": stages, "current_stage": "training", "next_action": "Start a training run when you are ready."}
    if status == ProjectStatus.RUNNING.value:
        return {**state, "stages": stages, "current_stage": "training", "next_action": "Training is in progress; poll the run endpoint for completion."}
    if status == ProjectStatus.FAILED.value:
        return {**state, "stages": stages, "current_stage": "training", "next_action": "Review the failure and start a new run after correcting the issue."}
    if status == ProjectStatus.COMPLETED.value:
        return {**state, "stages": stages, "current_stage": "report", "next_action": "Review the completed report and model leaderboard."}
    return {**state, "stages": stages}
