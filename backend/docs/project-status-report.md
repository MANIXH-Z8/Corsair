# AutoBuild backend: complete implementation status report

**Report date:** 2026-09-22  
**Repository:** `C:/Users/Manish D/Documents/Corsair`  
**Current branch when written:** `codex/durable-job-observability`  
**Baseline commit covered by this report:** `c780752 Add durable run events and recovery`

## 1. Product goal

AutoBuild is an agent-assisted tabular AutoML MVP for non-technical users. A user should be able to describe a business prediction problem in plain language, refine and approve a machine-readable brief, upload historical CSV/XLSX data, inspect data-readiness feedback, compare appropriate baseline models, receive a report, download the fitted model, and submit new records for prediction.

This MVP supports only:

- Tabular **classification**.
- Tabular **regression**.
- CSV and XLSX uploads up to 25 MB.
- Local execution with a development API key.

It explicitly does not yet support forecasting, clustering, NLP, computer vision, data connectors, multi-tenancy, live production serving, or real user identity/authentication.

## 2. Current backend workflow

```text
Plain-language problem
  → LangGraph discovery and local RAG recommendations
  → user clarifies and explicitly approves task/target/metric
  → CSV/XLSX upload and data-readiness profile
  → deterministic run plan
  → queued/running model comparison
  → report, model card, artifact download, and batch prediction
```

Project states are:

```text
discovery → awaiting_approval → awaiting_data → data_ready
                                    ↓
                              blocked (data issue)

data_ready → running → completed | failed
```

## 3. Architecture and technology

| Area | Current implementation |
|---|---|
| API | FastAPI 0.135.1 |
| Agent/workflow boundary | LangGraph 1.0.8 `StateGraph` |
| Retrieval | Deterministic local RAG-style ranking over reviewed use cases in `app/knowledge/use_cases.json` |
| Database | SQLite |
| Files | Local upload, template, and joblib artifact directories under `backend/data` |
| Training | scikit-learn 1.7.2 pipelines and cross-validation |
| Background execution | Bounded in-process `ThreadPoolExecutor` (two workers) |
| Tests | pytest + FastAPI `TestClient` |
| Supported Python runtime | Python 3.10+; validated with Python 3.10.11 |

The main modules are:

| Module | Responsibility |
|---|---|
| `app/main.py` | HTTP routes, state gates, job scheduling, report/inference endpoints |
| `app/orchestration.py` | LangGraph discovery and lifecycle-summary graphs |
| `app/rag.py` | Local reviewed-use-case retrieval |
| `app/discovery.py` | Deterministic extraction/refinement of task type and target column |
| `app/ml.py` | Dataset profile, candidate training, cross-validation, model card generation |
| `app/planning.py` | Candidate/metric/validation plan before a run |
| `app/inference.py` | Safe completed-model batch prediction |
| `app/db.py` | SQLite schema, JSON storage, workflow events, interrupted-job recovery |
| `app/config.py` | Environment, API key, data paths, upload limit, CORS allow-list |

## 4. Agentic and RAG implementation

### LangGraph

LangGraph is used for the discovery/refinement path and lifecycle presentation:

- Initial problem statement → inferred project specification.
- Follow-up messages → refined specification.
- Specification validation → either clarification required or approval ready.
- Persisted project/run state → frontend-friendly workflow stage and next action.

The graph is intentionally deterministic at this stage. It is a real workflow boundary rather than an unbounded autonomous coding agent. This is appropriate for the MVP because model selection and training must be reproducible and auditable.

### Local RAG-style retrieval

`app/rag.py` ranks reviewed records from `app/knowledge/use_cases.json` using task-aware text overlap. It returns up to three recommendations containing use-case ID, task type, suggested fields, and model notes.

This is not a hosted vector database or web retrieval system. It is deliberately local and reviewed so the MVP does not present unsupported or unverified algorithm advice as fact. A vector store can replace this implementation later without changing the discovery API contract.

## 5. Data readiness and model training

### Data profiling

After an approved specification, the upload endpoint profiles the dataset and returns:

- Rows, columns, field names, types, missing-value counts, duplicate-row count.
- Target distribution.
- Feature-quality groups: usable, empty, and constant columns.
- **Blockers** that prevent training.
- **Warnings** that should be shown but do not prevent training.

Current blockers include:

- Empty file or fewer than 30 rows.
- Missing or absent target column.
- Unapproved task type.
- Target with fewer than two non-empty values.
- Classification class with fewer than two labeled rows.
- Non-numeric regression target.
- No feature columns, or no usable non-empty/non-constant feature columns.

Current warnings include missing target rows, duplicate records, fully empty columns, and constant columns.

### Candidate models

| Task | Candidates | Primary metric | Validation |
|---|---|---|---|
| Classification | Logistic Regression, Random Forest, Histogram Gradient Boosting | F1 (weighted internally) | Stratified cross-validation, up to 5 folds |
| Regression | Ridge, Random Forest, Histogram Gradient Boosting | MAE | Cross-validation, up to 5 folds |

Every candidate is wrapped in a scikit-learn pipeline. Numeric fields use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding. The preprocessing is fit inside validation folds, avoiding that class of leakage.

### Run plan

`GET /projects/{project_id}/run-plan` exposes the exact planned candidates, the reason for each, target, metric direction, validation strategy, dataset summary when uploaded, and reproducibility settings (random seed `42`).

## 6. Report, explainability, artifacts, and inference

### Report and model card

`GET /projects/{project_id}/report` returns:

- Project and approved specification.
- Dataset profile.
- Completed run and ordered leaderboard.
- Model card.
- High-level limitations.

The model card includes the winning model, task type, metric, validation method/folds/scoring, data summary, random seed, limitations, and top feature impacts.

Feature impact is measured using three-repeat permutation importance after fitting the winning pipeline. It is labelled **exploratory**, calculated on the training data, non-causal, and sensitive to correlated fields. The frontend must never present it as a causal explanation or a fairness result.

### Artifacts

Completed models are saved as server-created `.joblib` artifacts. `GET /runs/{run_id}/artifact` only serves an artifact for a completed run. The backend derives the artifact name from the stored run result and strips path components, so an HTTP caller cannot choose an arbitrary server file.

### Batch prediction

`POST /runs/{run_id}/predict` accepts up to 500 records. It requires a completed run and every feature captured at training time. It returns predictions and, for classifiers that support it, per-class probabilities. Missing required fields return `422` before entering the model pipeline.

## 7. Durable job handling and observability

The local executor is not a production queue, but the MVP now avoids silent/stuck job states:

- Queueing, start, completion, and failure are persisted in the `run_events` table.
- State transition and its corresponding run event commit in the same SQLite transaction.
- A database-conditional transition prevents two active runs being started for the same project.
- `GET /runs/{run_id}/events` exposes the durable timeline.
- On server startup, queued/running jobs left from a prior local process are marked `failed` with a retryable restart-interruption message.
- Failure messages expose the failure category (for example, `ValueError`) but do not expose raw internal exception content.

For a hosted deployment, replace the local `ThreadPoolExecutor` with durable queue workers before claiming high availability.

## 8. Complete API inventory

All routes except `/health` and `/ready` require the `X-API-Key` header.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness metadata and API version |
| GET | `/ready` | SQLite/storage readiness check |
| POST | `/projects` | Create a project from a plain-language problem |
| GET | `/projects/{project_id}` | Fetch project specification and status |
| POST | `/projects/{project_id}/messages` | Refine a discovery-stage project brief |
| POST | `/projects/{project_id}/approve-spec` | Approve task type, target field, and metric |
| POST | `/projects/{project_id}/datasets` | Upload/profile a CSV or XLSX dataset |
| GET | `/projects/{project_id}/data-template` | Download a target-aware CSV template |
| GET | `/projects/{project_id}/run-plan` | Retrieve planned candidates/metric/validation |
| POST | `/projects/{project_id}/runs` | Queue a validated training run |
| GET | `/runs/{run_id}` | Poll current run state/result |
| GET | `/runs/{run_id}/events` | Fetch durable run timeline |
| GET | `/projects/{project_id}/runs` | Fetch project run history |
| GET | `/runs/{run_id}/artifact` | Download a completed fitted model |
| POST | `/runs/{run_id}/predict` | Perform batch inference with a completed model |
| GET | `/projects/{project_id}/report` | Retrieve completed project report/model card |
| GET | `/projects/{project_id}/frontend-contract` | Frontend statuses, screens, auth/CORS contract |
| GET | `/projects/{project_id}/workflow` | Stepper stage, next action, and workflow events |

## 9. Browser/frontend integration contract

The frontend should:

1. Create a project and render `spec.missing_information` as guided questions.
2. Render local RAG `spec.recommendations` as optional guidance, not guarantees.
3. Require explicit spec approval before enabling upload.
4. Show profile blockers and warnings separately after upload.
5. Display `run-plan` before the user starts training.
6. Poll `/runs/{run_id}` until `completed` or `failed`; use `/runs/{run_id}/events` for the timeline.
7. Render the leaderboard and model card with the explainability disclaimer.
8. Offer artifact download only when `artifact_available` is true.
9. Use `/predict` for an individual/batch prediction screen.

For browser access, configure `AUTOBUILD_CORS_ORIGINS` with explicit comma-separated frontend origins. Defaults are `http://localhost:3000` and `http://localhost:5173`. Do not use an unrestricted wildcard allow-list with this API-key model.

The detailed frontend instructions are also maintained in `backend/docs/frontend-handoff.md`.

## 10. Runtime configuration and local startup

Reference settings are in `backend/.env.example`. Set these as process environment variables:

```text
AUTOBUILD_ENVIRONMENT=development|test|production
AUTOBUILD_API_KEY=<long-random-secret>
AUTOBUILD_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
AUTOBUILD_DATA_DIR=<optional absolute data directory>
```

The application refuses to start in `production` if the development API key is still configured. Runtime files live under `backend/data` by default and are intentionally ignored by Git.

Local test command:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
```

Latest result: **7 passed**. The only known non-failing warnings are from Starlette/AnyIO deprecation behavior and joblib's Windows physical-core detection fallback.

## 11. Git history and completed phases

The repository has been initialized locally. No remote is configured and **nothing has been pushed to GitHub**.

| Commit | Branch/milestone | What it added |
|---|---|---|
| `26929de` | `main` baseline | LangGraph discovery, local RAG retrieval, Python 3.10 compatibility, initial docs/tests |
| `ccfb9ae` | `develop` | Git workflow documentation |
| `5660a82` + `0d464b2` | workflow readiness | Approval/data lifecycle guards, workflow endpoint, event timeline report |
| `b03aa05` | run planning | Deterministic candidate/metric/validation plan |
| `9fd16e6` | data readiness | Training blockers and data-quality warnings |
| `0a00ad1` | artifacts | Artifact download and run history |
| `d0161e8` | inference | Completed-model prediction endpoint |
| `4246891` | operational config | CORS, environment safety, health/readiness, environment template |
| `2b9c70a` | explainability | Model card and exploratory feature impact |
| `c780752` | durable jobs | Atomic run events, duplicate-run guard, restart reconciliation |

Current local branches are `main`, `develop`, and the stacked `codex/*` implementation branches. The current branch contains all completed work, but it has not yet been merged into `develop` or `main`.

The separate command-level record is `backend/docs/git-workflow-report.md`.

## 12. Known limitations and honest production-readiness assessment

The backend is a strong, tested **local MVP**, not yet a hosted production service. Before a real production release, the following are required:

- PostgreSQL plus database migrations instead of SQLite.
- Durable task queue/workers (for example, a managed queue and isolated workers) instead of in-process threads.
- Object storage, retention policies, malware scanning, file quotas, and cleanup jobs for uploads/artifacts.
- Real authentication/authorization, tenant/project ownership, secret management, HTTPS, rate limits, and audit logging.
- Structured application logging, metrics, tracing, alerting, backups, disaster recovery, and deployment automation.
- CI, dependency/security scanning, container build, integration/load tests, and operational runbooks.
- Model governance: fairness checks, drift monitoring, data lineage, approval/access controls, and a proper deployment/serving strategy.
- More robust RAG infrastructure if reviewed local-use-case retrieval is no longer sufficient.

These limitations are documented so the MVP is not overrepresented as a production AutoML platform.

## 13. Scope completed versus next decision

The two final backend phases requested after the earlier implementation plan—**model report/explainability** and **durable job execution/observability**—are complete.

The next product decision is whether to:

1. Review and merge the completed local branches into `develop`, then start the frontend using the documented APIs; or
2. Expand the backend into a deployment-grade platform, which is a separate infrastructure/security programme rather than a small MVP phase.
