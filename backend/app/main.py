from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import db
from .config import API_KEY, ARTIFACT_DIR, MAX_UPLOAD_BYTES, TEMPLATE_DIR, UPLOAD_DIR
from .inference import predict_records
from .ml import profile_dataset, train
from .orchestration import run_followup_discovery, run_initial_discovery, workflow_summary
from .planning import build_run_plan
from .schemas import ApprovalRequest, CreateProjectRequest, DatasetResponse, MessageRequest, PredictionRequest, PredictionResponse, ProjectResponse, ProjectSpec, ProjectStatus, RunPlanResponse, RunResponse, WorkflowEvent, WorkflowResponse

app = FastAPI(title="AutoBuild Backend", version="0.1.0")
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="automl")
db.initialize()


def require_api_key(request: Request):
    if request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def project_or_404(project_id: str):
    row = db.fetch_one("SELECT * FROM projects WHERE id = ?", (project_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def project_response(row) -> ProjectResponse:
    return ProjectResponse(id=row["id"], name=row["name"], status=row["status"], problem_statement=row["problem_statement"], spec=ProjectSpec(**db.load(row["spec_json"])))


def run_response(row) -> RunResponse:
    result = db.load(row["result_json"])
    artifact_available = artifact_path_from_run(row) is not None
    return RunResponse(
        id=row["id"], project_id=row["project_id"], status=row["status"], result=result, error=row["error"],
        created_at=row["created_at"], updated_at=row["updated_at"], artifact_available=artifact_available,
    )


def artifact_path_from_run(row) -> Path | None:
    result = db.load(row["result_json"], {}) or {}
    artifact_name = Path(str(result.get("artifact", ""))).name
    artifact_path = ARTIFACT_DIR / artifact_name
    return artifact_path if artifact_name and artifact_path.is_file() else None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/projects", response_model=ProjectResponse, dependencies=[Depends(require_api_key)])
def create_project(payload: CreateProjectRequest):
    project_id, spec = str(uuid.uuid4()), run_initial_discovery(payload.problem_statement)
    status = ProjectStatus.AWAITING_APPROVAL if spec.readiness == "ready_for_approval" else ProjectStatus.DISCOVERY
    with db.connection() as conn:
        conn.execute("INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)", (project_id, payload.name, payload.problem_statement, status, db.dump(spec.model_dump()), db.now()))
        conn.execute("INSERT INTO messages (project_id, role, content, created_at) VALUES (?, ?, ?, ?)", (project_id, "user", payload.problem_statement, db.now()))
    db.record_workflow_event(project_id, "discovery_started", "Problem statement captured and initial project brief generated.")
    return ProjectResponse(id=project_id, name=payload.name, status=status, problem_statement=payload.problem_statement, spec=spec)


@app.get("/projects/{project_id}", response_model=ProjectResponse, dependencies=[Depends(require_api_key)])
def get_project(project_id: str):
    return project_response(project_or_404(project_id))


@app.post("/projects/{project_id}/messages", response_model=ProjectResponse, dependencies=[Depends(require_api_key)])
def add_message(project_id: str, payload: MessageRequest):
    row = project_or_404(project_id)
    if row["status"] not in {ProjectStatus.DISCOVERY.value, ProjectStatus.AWAITING_APPROVAL.value}:
        raise HTTPException(status_code=409, detail="The project brief is already approved; create a new project to change its scope.")
    spec = run_followup_discovery(db.load(row["spec_json"]), payload.content)
    status = ProjectStatus.AWAITING_APPROVAL if spec.readiness == "ready_for_approval" else ProjectStatus.DISCOVERY
    with db.connection() as conn:
        conn.execute("INSERT INTO messages (project_id, role, content, created_at) VALUES (?, ?, ?, ?)", (project_id, "user", payload.content, db.now()))
        conn.execute("UPDATE projects SET spec_json = ?, status = ? WHERE id = ?", (db.dump(spec.model_dump()), status, project_id))
    db.record_workflow_event(project_id, "discovery_refined", "Project brief updated from a user message.")
    return project_response(project_or_404(project_id))


@app.post("/projects/{project_id}/approve-spec", response_model=ProjectResponse, dependencies=[Depends(require_api_key)])
def approve_spec(project_id: str, payload: ApprovalRequest):
    row = project_or_404(project_id)
    spec = db.load(row["spec_json"])
    spec.update({"task_type": payload.task_type, "target_column": payload.target_column, "primary_metric": payload.primary_metric or ("f1" if payload.task_type == "classification" else "mae"), "missing_information": [], "readiness": "approved"})
    with db.connection() as conn:
        conn.execute("UPDATE projects SET spec_json = ?, status = ? WHERE id = ?", (db.dump(spec), ProjectStatus.AWAITING_DATA, project_id))
    db.record_workflow_event(project_id, "spec_approved", "Task type, target column, and metric approved; awaiting historical data.")
    return project_response(project_or_404(project_id))


@app.post("/projects/{project_id}/datasets", response_model=DatasetResponse, dependencies=[Depends(require_api_key)])
def upload_dataset(project_id: str, file: UploadFile = File(...)):
    row = project_or_404(project_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=415, detail="Only .csv and .xlsx files are supported")
    if row["status"] not in {ProjectStatus.AWAITING_DATA.value, ProjectStatus.BLOCKED.value, ProjectStatus.DATA_READY.value}:
        raise HTTPException(status_code=409, detail="Approve the project brief before uploading data")
    dataset_id, target_path = str(uuid.uuid4()), UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with target_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    if target_path.stat().st_size > MAX_UPLOAD_BYTES:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Upload exceeds 25 MB limit")
    spec = db.load(row["spec_json"])
    try:
        profile, blockers = profile_dataset(target_path, spec.get("target_column"), spec.get("task_type"))
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse dataset: {exc}") from exc
    with db.connection() as conn:
        conn.execute("INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (dataset_id, project_id, file.filename, str(target_path), db.dump(profile), not blockers, db.dump(blockers), db.now()))
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", (ProjectStatus.DATA_READY if not blockers else ProjectStatus.BLOCKED, project_id))
    db.record_workflow_event(project_id, "data_profiled", "Dataset is training-ready." if not blockers else "Dataset has blockers that must be resolved.")
    return DatasetResponse(id=dataset_id, project_id=project_id, filename=file.filename or "upload", profile=profile, training_ready=not blockers, blockers=blockers, warnings=profile["warnings"])


@app.get("/projects/{project_id}/data-template", dependencies=[Depends(require_api_key)])
def download_data_template(project_id: str):
    row = project_or_404(project_id)
    spec = db.load(row["spec_json"])
    target = spec.get("target_column") or "target"
    template_path = TEMPLATE_DIR / f"{project_id}.csv"
    template_path.write_text(f"entity_id,feature_1,feature_2,{target}\nexample-001,,,\n", encoding="utf-8")
    return FileResponse(template_path, media_type="text/csv", filename=f"{row['name']}-data-template.csv")


@app.post("/projects/{project_id}/runs", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def start_run(project_id: str):
    row = project_or_404(project_id)
    dataset = db.fetch_one("SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,))
    if not dataset:
        raise HTTPException(status_code=409, detail="Upload a dataset before starting a run")
    if row["status"] != ProjectStatus.DATA_READY.value:
        raise HTTPException(status_code=409, detail="Resolve data blockers before starting a run")
    if not dataset["training_ready"]:
        raise HTTPException(status_code=409, detail="Dataset is not ready for training")
    run_id = str(uuid.uuid4())
    with db.connection() as conn:
        conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (run_id, project_id, dataset["id"], "queued", None, None, db.now(), db.now()))
        conn.execute("UPDATE projects SET status = ? WHERE id = ?", (ProjectStatus.RUNNING, project_id))
    db.record_workflow_event(project_id, "run_queued", "Training run queued.")
    executor.submit(_execute_run, run_id, row["spec_json"], dataset["stored_path"])
    return RunResponse(id=run_id, project_id=project_id, status="queued")


@app.get("/projects/{project_id}/run-plan", response_model=RunPlanResponse, dependencies=[Depends(require_api_key)])
def get_run_plan(project_id: str):
    row = project_or_404(project_id)
    dataset = db.fetch_one("SELECT profile_json FROM datasets WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,))
    try:
        plan = build_run_plan(db.load(row["spec_json"]), db.load(dataset["profile_json"]) if dataset else None)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunPlanResponse(**plan)


def _execute_run(run_id: str, spec_json: str, dataset_path: str):
    spec = db.load(spec_json)
    with db.connection() as conn:
        conn.execute("UPDATE runs SET status = ?, updated_at = ? WHERE id = ?", ("running", db.now(), run_id))
    try:
        result = train(Path(dataset_path), spec["target_column"], spec["task_type"], spec["primary_metric"], ARTIFACT_DIR / f"{run_id}.joblib")
        with db.connection() as conn:
            conn.execute("UPDATE runs SET status = ?, result_json = ?, updated_at = ? WHERE id = ?", ("completed", db.dump(result), db.now(), run_id))
            conn.execute("UPDATE projects SET status = ? WHERE id = (SELECT project_id FROM runs WHERE id = ?)", (ProjectStatus.COMPLETED, run_id))
            project_id = conn.execute("SELECT project_id FROM runs WHERE id = ?", (run_id,)).fetchone()["project_id"]
        db.record_workflow_event(project_id, "run_completed", "Training completed and the leaderboard is available.")
    except Exception as exc:
        with db.connection() as conn:
            conn.execute("UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE id = ?", ("failed", "Training failed; inspect server logs for details.", db.now(), run_id))
            conn.execute("UPDATE projects SET status = ? WHERE id = (SELECT project_id FROM runs WHERE id = ?)", (ProjectStatus.FAILED, run_id))
            project_id = conn.execute("SELECT project_id FROM runs WHERE id = ?", (run_id,)).fetchone()["project_id"]
        db.record_workflow_event(project_id, "run_failed", "Training failed; inspect server logs before retrying.")


@app.get("/runs/{run_id}", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def get_run(run_id: str):
    row = db.fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_response(row)


@app.get("/projects/{project_id}/runs", response_model=list[RunResponse], dependencies=[Depends(require_api_key)])
def list_project_runs(project_id: str):
    project_or_404(project_id)
    rows = db.fetch_all("SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
    return [run_response(row) for row in rows]


@app.get("/runs/{run_id}/artifact", dependencies=[Depends(require_api_key)])
def download_artifact(run_id: str):
    row = db.fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="A completed training run is required before the model artifact is available")
    artifact_path = artifact_path_from_run(row)
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="Model artifact is unavailable")
    return FileResponse(artifact_path, media_type="application/octet-stream", filename=f"{run_id}-model.joblib")


@app.post("/runs/{run_id}/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict_with_run(run_id: str, payload: PredictionRequest):
    row = db.fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="A completed training run is required before predictions are available")
    artifact_path = artifact_path_from_run(row)
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="Model artifact is unavailable")
    result = db.load(row["result_json"], {}) or {}
    try:
        predictions, probabilities = predict_records(artifact_path, result["features_used"], payload.records)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(run_id=run_id, model=result["best_model"], predictions=predictions, probabilities=probabilities)


@app.get("/projects/{project_id}/report", dependencies=[Depends(require_api_key)])
def get_report(project_id: str):
    project = project_or_404(project_id)
    run = db.fetch_one("SELECT * FROM runs WHERE project_id = ? AND status = 'completed' ORDER BY updated_at DESC LIMIT 1", (project_id,))
    if not run:
        raise HTTPException(status_code=409, detail="A completed training run is required before a report is available")
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],))
    return {
        "project": project_response(project), "dataset_profile": db.load(dataset["profile_json"]),
        "run": run_response(run),
        "limitations": ["Scores are cross-validation estimates, not a guarantee of production performance.", "The model is not deployed by this MVP."],
    }


@app.get("/projects/{project_id}/frontend-contract", dependencies=[Depends(require_api_key)])
def frontend_contract(project_id: str):
    project_or_404(project_id)
    return {"project_statuses": [status.value for status in ProjectStatus], "screens": ["discovery", "spec_approval", "data_upload", "data_readiness", "run_progress", "leaderboard", "report"], "workflow_endpoint": "/projects/{project_id}/workflow", "polling": {"endpoint": "/runs/{run_id}", "until": ["completed", "failed"]}}


@app.get("/projects/{project_id}/workflow", response_model=WorkflowResponse, dependencies=[Depends(require_api_key)])
def get_workflow(project_id: str):
    project = project_or_404(project_id)
    dataset = db.fetch_one("SELECT id FROM datasets WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,))
    completed_run = db.fetch_one("SELECT id FROM runs WHERE project_id = ? AND status = 'completed' ORDER BY updated_at DESC LIMIT 1", (project_id,))
    current_stage, next_action, stages = workflow_summary(project["status"], db.load(project["spec_json"]), bool(dataset), bool(completed_run))
    events = db.fetch_all("SELECT id, step, detail, created_at FROM workflow_events WHERE project_id = ? ORDER BY id DESC LIMIT 50", (project_id,))
    return WorkflowResponse(project_id=project_id, status=project["status"], current_stage=current_stage, next_action=next_action, stages=stages, events=[WorkflowEvent(**dict(event)) for event in events])
