# Frontend handoff

Send `X-API-Key` in local development. Render `spec.missing_information` as a guided discovery view, then require approval of task type, target column, and metric before upload. Render `spec.recommendations` as optional guidance cards; these are retrieved from reviewed local use cases, not guaranteed model choices.

After upload, show `profile`, `blockers`, and `training_ready`; link users to `GET /projects/{id}/data-template` when blocked. On run creation, poll `GET /runs/{run_id}` every 1–2 seconds until `completed` or `failed`. Render `result.leaderboard` in returned order: higher is better for F1 and lower is better for MAE. Use `GET /projects/{id}/report` for the results screen.

Before showing the run button, call `GET /projects/{id}/run-plan`. Render its candidates and reasons as the planned experiment, and use `metric_direction` to explain how the leaderboard is judged. The endpoint returns `409` until the user has supplied enough information to identify classification or regression.

Render `DatasetResponse.warnings` separately from blockers: warnings inform the user about data quality but do not prevent a run; blockers keep the project in `blocked` and must be resolved. The profile's `quality` object identifies usable, empty, and constant feature columns.

Use `GET /projects/{id}/runs` for run history. Each run exposes `artifact_available`; only when it is true should the UI offer the authenticated download at `GET /runs/{run_id}/artifact`.

For the MVP's real-time prediction screen, post up to 500 records to `POST /runs/{run_id}/predict` as `{ "records": [{...}] }`. Include every value from `run.result.features_used`; missing required columns return `422`. Classification returns probability maps keyed by class label when supported by the selected model.

For browser integration, configure the frontend origin in `AUTOBUILD_CORS_ORIGINS` (a comma-separated allow-list). The local defaults permit `http://localhost:3000` and `http://localhost:5173`. Send `X-API-Key` on every protected request. Use `/health` for liveness and `/ready` before declaring the API usable.

Render `report.model_card` as an explainability section: validation method, data summary, limitations, and `feature_impact.top_features`. Display the feature-impact scope statement verbatim or equivalently; it is exploratory and must not be presented as a causal explanation.

Use `GET /projects/{id}/workflow` to drive the stepper, current instruction (`next_action`), and activity timeline (`events`). The endpoint is authoritative: do not infer a screen from the client alone. A valid CSV/XLSX upload before the project reaches `awaiting_data` receives `409`; after approval, prevent edits to the discovery conversation and start a new project for a material scope change.

Use `GET /projects/{id}/frontend-contract` for statuses and expected screens.
