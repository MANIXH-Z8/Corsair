import io
import time

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
HEADERS = {"X-API-Key": "local-development-key"}


def test_classification_workflow():
    response = client.post("/projects", headers=HEADERS, json={"name": "Churn", "problem_statement": "Predict customer churn as a classification outcome."})
    assert response.status_code == 200
    project = response.json()
    project_id = project["id"]
    approval = client.post(f"/projects/{project_id}/approve-spec", headers=HEADERS, json={"task_type": "classification", "target_column": "churn"})
    assert approval.status_code == 200
    frame = pd.DataFrame({"age": list(range(60)), "plan": ["basic", "pro"] * 30, "churn": [0, 1] * 30})
    upload = client.post(f"/projects/{project_id}/datasets", headers=HEADERS, files={"file": ("churn.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert upload.status_code == 200
    assert upload.json()["training_ready"] is True
    run = client.post(f"/projects/{project_id}/runs", headers=HEADERS)
    assert run.status_code == 200
    for _ in range(100):
        result = client.get(f"/runs/{run.json()['id']}", headers=HEADERS).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert result["status"] == "completed"
    assert result["result"]["best_model"]
    assert result["result"]["model_card"]["feature_impact"]["method"] == "permutation_importance_on_training_data"
    assert result["artifact_available"] is True
    history = client.get(f"/projects/{project_id}/runs", headers=HEADERS)
    assert history.status_code == 200
    assert history.json()[0]["id"] == run.json()["id"]
    artifact = client.get(f"/runs/{run.json()['id']}/artifact", headers=HEADERS)
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "application/octet-stream"
    prediction = client.post(f"/runs/{run.json()['id']}/predict", headers=HEADERS, json={"records": [{"age": 28, "plan": "basic"}, {"age": 54, "plan": "pro"}]})
    assert prediction.status_code == 200
    assert len(prediction.json()["predictions"]) == 2
    assert prediction.json()["probabilities"] is not None
    invalid_prediction = client.post(f"/runs/{run.json()['id']}/predict", headers=HEADERS, json={"records": [{"age": 28}]})
    assert invalid_prediction.status_code == 422


def test_operational_endpoints_and_frontend_cors_contract():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"
    preflight = client.options("/projects", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_rejects_unsupported_file():
    project = client.post("/projects", headers=HEADERS, json={"name": "Price", "problem_statement": "Predict house price using regression."}).json()
    response = client.post(f"/projects/{project['id']}/datasets", headers=HEADERS, files={"file": ("bad.txt", b"no", "text/plain")})
    assert response.status_code == 415


def test_data_profile_blocks_invalid_targets_and_explains_feature_quality():
    project = client.post("/projects", headers=HEADERS, json={"name": "Refund", "problem_statement": "Predict refund amount using regression."}).json()
    project_id = project["id"]
    assert client.post(f"/projects/{project_id}/approve-spec", headers=HEADERS, json={"task_type": "regression", "target_column": "refund"}).status_code == 200
    frame = pd.DataFrame({"empty_feature": [None] * 30, "same_value": [1] * 30, "refund": ["high"] * 30})
    response = client.post(f"/projects/{project_id}/datasets", headers=HEADERS, files={"file": ("refunds.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert data["training_ready"] is False
    assert any("regression target must be numeric" in blocker.lower() for blocker in data["blockers"])
    assert any("non-empty feature column" in blocker.lower() for blocker in data["blockers"])
    assert data["warnings"]


def test_langgraph_discovery_refines_project_spec():
    response = client.post("/projects", headers=HEADERS, json={"name": "Delivery", "problem_statement": "I need to reduce delivery delay risk for orders."})
    assert response.status_code == 200
    project = response.json()
    assert project["spec"]["recommendations"]
    assert project["status"] == "discovery"

    update = client.post(
        f"/projects/{project['id']}/messages",
        headers=HEADERS,
        json={"content": "This is classification and the target column is delayed"},
    )
    assert update.status_code == 200
    updated_project = update.json()
    assert updated_project["status"] == "awaiting_approval"
    assert updated_project["spec"]["target_column"] == "delayed"
    assert updated_project["spec"]["task_type"] == "classification"


def test_workflow_requires_approval_then_exposes_frontend_state():
    project = client.post(
        "/projects",
        headers=HEADERS,
        json={"name": "Claims", "problem_statement": "Predict whether an insurance claim is fraudulent; target column is fraud."},
    ).json()
    project_id = project["id"]
    workflow = client.get(f"/projects/{project_id}/workflow", headers=HEADERS)
    assert workflow.status_code == 200
    assert workflow.json()["current_stage"] == "approval"

    frame = pd.DataFrame({"claim_value": [100, 200], "fraud": [0, 1]})
    unapproved_upload = client.post(f"/projects/{project_id}/datasets", headers=HEADERS, files={"file": ("claims.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert unapproved_upload.status_code == 409

    approval = client.post(f"/projects/{project_id}/approve-spec", headers=HEADERS, json={"task_type": "classification", "target_column": "fraud"})
    assert approval.json()["status"] == "awaiting_data"
    awaiting_data = client.get(f"/projects/{project_id}/workflow", headers=HEADERS).json()
    assert awaiting_data["current_stage"] == "data_upload"
    assert awaiting_data["events"][0]["step"] == "spec_approved"

    plan_before_data = client.get(f"/projects/{project_id}/run-plan", headers=HEADERS)
    assert plan_before_data.status_code == 200
    assert [candidate["name"] for candidate in plan_before_data.json()["candidates"]] == ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    assert plan_before_data.json()["metric_direction"] == "higher_is_better"


def test_regression_workflow_and_report():
    project = client.post("/projects", headers=HEADERS, json={"name": "Property price", "problem_statement": "Predict house price with regression."}).json()
    project_id = project["id"]
    assert client.post(f"/projects/{project_id}/approve-spec", headers=HEADERS, json={"task_type": "regression", "target_column": "price"}).status_code == 200
    frame = pd.DataFrame({"size": list(range(40, 100)), "rooms": [1, 2, 3] * 20, "price": [100000 + n * 3000 for n in range(60)]})
    upload = client.post(f"/projects/{project_id}/datasets", headers=HEADERS, files={"file": ("prices.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert upload.json()["training_ready"] is True
    run = client.post(f"/projects/{project_id}/runs", headers=HEADERS).json()
    for _ in range(100):
        result = client.get(f"/runs/{run['id']}", headers=HEADERS).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert result["status"] == "completed"
    report = client.get(f"/projects/{project_id}/report", headers=HEADERS)
    assert report.status_code == 200
    assert report.json()["run"]["result"]["primary_metric"] == "mae"
    assert report.json()["model_card"]["validation"]["method"] == "cross_validation"
