# Frontend handoff

Send `X-API-Key` in local development. Render `spec.missing_information` as a guided discovery view, then require approval of task type, target column, and metric before upload. Render `spec.recommendations` as optional guidance cards; these are retrieved from reviewed local use cases, not guaranteed model choices.

After upload, show `profile`, `blockers`, and `training_ready`; link users to `GET /projects/{id}/data-template` when blocked. On run creation, poll `GET /runs/{run_id}` every 1–2 seconds until `completed` or `failed`. Render `result.leaderboard` in returned order: higher is better for F1 and lower is better for MAE. Use `GET /projects/{id}/report` for the results screen.

Use `GET /projects/{id}/frontend-contract` for statuses and expected screens.
