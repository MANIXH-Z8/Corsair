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
    for _ in range(50):
        result = client.get(f"/runs/{run.json()['id']}", headers=HEADERS).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert result["status"] == "completed"
    assert result["result"]["best_model"]


def test_rejects_unsupported_file():
    project = client.post("/projects", headers=HEADERS, json={"name": "Price", "problem_statement": "Predict house price using regression."}).json()
    response = client.post(f"/projects/{project['id']}/datasets", headers=HEADERS, files={"file": ("bad.txt", b"no", "text/plain")})
    assert response.status_code == 415


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


def test_regression_workflow_and_report():
    project = client.post("/projects", headers=HEADERS, json={"name": "Property price", "problem_statement": "Predict house price with regression."}).json()
    project_id = project["id"]
    assert client.post(f"/projects/{project_id}/approve-spec", headers=HEADERS, json={"task_type": "regression", "target_column": "price"}).status_code == 200
    frame = pd.DataFrame({"size": list(range(40, 100)), "rooms": [1, 2, 3] * 20, "price": [100000 + n * 3000 for n in range(60)]})
    upload = client.post(f"/projects/{project_id}/datasets", headers=HEADERS, files={"file": ("prices.csv", frame.to_csv(index=False).encode(), "text/csv")})
    assert upload.json()["training_ready"] is True
    run = client.post(f"/projects/{project_id}/runs", headers=HEADERS).json()
    for _ in range(50):
        result = client.get(f"/runs/{run['id']}", headers=HEADERS).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert result["status"] == "completed"
    report = client.get(f"/projects/{project_id}/report", headers=HEADERS)
    assert report.status_code == 200
    assert report.json()["run"]["result"]["primary_metric"] == "mae"
