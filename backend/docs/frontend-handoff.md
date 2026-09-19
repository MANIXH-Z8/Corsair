# Frontend handoff

Send `X-API-Key` in local development. Render `spec.missing_information` as a guided discovery view, then require approval of task type, target column, and metric before upload. Render `spec.recommendations` as optional guidance cards; these are retrieved from reviewed local use cases, not guaranteed model choices.

After upload, show `profile`, `blockers`, and `training_ready`; link users to `GET /projects/{id}/data-template` when blocked. On run creation, poll `GET /runs/{run_id}` every 1–2 seconds until `completed` or `failed`. Render `result.leaderboard` in returned order: higher is better for F1 and lower is better for MAE. Use `GET /projects/{id}/report` for the results screen.

Before showing the run button, call `GET /projects/{id}/run-plan`. Render its candidates and reasons as the planned experiment, and use `metric_direction` to explain how the leaderboard is judged. The endpoint returns `409` until the user has supplied enough information to identify classification or regression.

Use `GET /projects/{id}/workflow` to drive the stepper, current instruction (`next_action`), and activity timeline (`events`). The endpoint is authoritative: do not infer a screen from the client alone. A valid CSV/XLSX upload before the project reaches `awaiting_data` receives `409`; after approval, prevent edits to the discovery conversation and start a new project for a material scope change.

Use `GET /projects/{id}/frontend-contract` for statuses and expected screens.
