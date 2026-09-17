from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from typing_extensions import TypedDict

from .discovery import apply_message, infer_spec
from .schemas import ProjectSpec

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


class _SequentialDiscoveryGraph:
    def invoke(self, state: DiscoveryState) -> DiscoveryState:
        state = _route_discovery(state)
        state = _validate_spec(state)
        return state


def run_initial_discovery(problem_statement: str) -> ProjectSpec:
    state = _compiled_graph().invoke({"operation": "create", "problem_statement": problem_statement})
    return ProjectSpec(**state["spec"])


def run_followup_discovery(current_spec: dict[str, Any], message: str) -> ProjectSpec:
    state = _compiled_graph().invoke({"operation": "refine", "current_spec": current_spec, "message": message})
    return ProjectSpec(**state["spec"])


@lru_cache(maxsize=1)
def _compiled_graph():
    if StateGraph is None:
        return _SequentialDiscoveryGraph()

    workflow = StateGraph(DiscoveryState)
    workflow.add_node("route_discovery", _route_discovery)
    workflow.add_node("validate_spec", _validate_spec)
    workflow.add_edge(START, "route_discovery")
    workflow.add_edge("route_discovery", "validate_spec")
    workflow.add_edge("validate_spec", END)
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
