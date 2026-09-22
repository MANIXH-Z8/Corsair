# Backend architecture

FastAPI exposes a protected REST API. SQLite stores project state, messages, dataset metadata, and run state. Source files and fitted joblib pipelines live in `backend/data`, which must be backed up and excluded from source control. A bounded two-thread local executor runs scikit-learn training.

Discovery and specification refinement run through `app/orchestration.py`. The orchestration boundary is implemented as a LangGraph `StateGraph` with explicit state validation, so future nodes can be added for richer RAG retrieval, human approval, data profiling, and run planning without spreading workflow logic across API handlers.

The first RAG layer is local and deterministic: `app/rag.py` retrieves reviewed use-case records from `app/knowledge/use_cases.json` and attaches up to three matching recommendations to the project spec. This avoids claiming unsupported intelligence while preserving a production path toward vector retrieval later.

Approved specifications are converted into a deterministic run plan by `app/planning.py`. The plan names the three candidate estimators, metric direction, validation strategy, and reproducibility controls before any training job is queued.

Dataset profiling is also a hard pre-flight gate. It blocks invalid target types, too-small class support, and unusable feature sets before the executor can start; non-blocking quality findings (missing targets, duplicates, empty or constant fields) are returned as warnings for the UI.

Completed joblib artifacts remain in the local artifact directory and are served only through a run-specific authenticated endpoint. The service derives the filename from a database result and discards path components before opening it, so request input cannot select arbitrary files.

The same guarded artifact lookup powers batch inference. The prediction endpoint accepts up to 500 tabular records, verifies that every training feature is present, and returns predictions (plus class probabilities when the fitted estimator supports them).

Runtime configuration is environment-based. CORS uses an explicit origin allow-list, the API key must be overridden in production, `/health` exposes liveness metadata, and `/ready` verifies SQLite access plus required storage directories. See `backend/.env.example` for the supported variables.

Workflow: `discovery → approval → awaiting_data → upload/profile → data_ready → queued/running → completed|failed`. The LangGraph lifecycle graph turns persisted project state into a current stage, next action, and frontend stage list. The API records immutable workflow events in SQLite, giving the future frontend an auditable user-facing timeline. Every candidate uses a scikit-learn pipeline, ensuring imputation, encoding, and scaling are fit within cross-validation folds.

Before external hosting, replace SQLite and the local executor with PostgreSQL plus isolated queue workers, then add compute controls, scanning, object storage, retention jobs, real identities, and observability.
