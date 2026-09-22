# Git workflow report

## Current branch strategy

This project uses a simple production-minded flow:

- `main` stores stable, tested code.
- `develop` will be used as the integration branch after the initial backend state is committed.
- `codex/*` branches are used for each implementation phase in this Codex workspace, then merged into `develop`.

For this phase, the repository did not have any commits yet, so the first commit records the backend state plus the LangGraph/RAG orchestration phase together.

## Commands used and why

```bash
git status --short --branch
```

Purpose: checked the current branch and pending changes before staging.

Why: this confirmed that only `.gitignore`, `.vscode/`, and `backend/` were untracked. `.vscode/` was intentionally left out of the commit.

```bash
git init
```

Purpose: created a local Git repository in `C:/Users/Manish D/Documents/Corsair`.

Why: Git cannot track changes, create branches, or commit project history until the repository is initialized.

```bash
git branch -m master main
```

Purpose: renamed the default initial branch from `master` to `main`.

Why: `main` is the project's stable branch name. It will contain production-ready backend milestones.

In this Codex environment, some Git commands used this form:

```bash
git -c safe.directory="C:/Users/Manish D/Documents/Corsair" <command>
```

Purpose: allowed Git to operate on this working tree despite Windows sandbox ownership differences.

Why: the repository was initialized from inside the Codex sandbox, while normal Git commands may run as your Windows user. Git protects against suspicious ownership changes, so the safe-directory override tells Git this project path is trusted for that command.

```bash
git status --short --branch
```

Purpose: checked the current branch and pending changes.

Why: this confirms what will be committed and prevents accidentally tracking unrelated files.

```bash
git add .gitignore backend
```

Purpose: stages the root ignore file and backend implementation/docs/tests.

Why: `.vscode/` is intentionally not staged yet. Editor settings should only be committed if we decide they are project-standard settings.

```bash
git commit -m "Add LangGraph orchestration and local RAG retrieval"
```

Purpose: creates a permanent checkpoint for the completed backend phase.

Why: this gives us a clean rollback/review point before starting the next phase.

Executed result:

```text
26929de Add LangGraph orchestration and local RAG retrieval
```

```bash
git switch -c develop
```

Purpose: created and switched to the `develop` branch after the first stable backend commit.

Why: future feature branches should be created from `develop`, then merged back into `develop` before anything reaches `main`.

## Phase covered by this commit

This commit covers the backend phase that introduced:

- LangGraph-based discovery orchestration.
- Local RAG-style retrieval from reviewed use cases.
- API wiring through the orchestration boundary.
- Python 3.10 compatibility fixes.
- Target-column parsing fixes.
- Backend documentation updates.
- Tests for discovery refinement.

## Verification

The backend test suite was run with:

```bash
..\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
```

Result:

```text
4 passed, 1 warning
```

The warning is from Starlette/AnyIO deprecation behavior and does not currently fail the backend workflow.

## Next Git steps after this commit

For this workflow-readiness phase, the following command was executed:

```bash
git switch -c codex/workflow-readiness
```

Purpose: created an isolated branch from `develop` before changing the lifecycle API.

Why: this keeps the integration branch reviewable while the phase is developed and tested.

For a later backend phase, follow:

```bash
git switch develop
git switch -c codex/<phase-name>
```

After a feature is tested:

```bash
git add .
git commit -m "<clear feature message>"
git switch develop
git merge --no-ff codex/<phase-name>
```

When `develop` is stable and ready for a release/demo:

```bash
git switch main
git merge --no-ff develop
```

## Workflow-readiness phase record

Implemented on `codex/workflow-readiness` and committed as:

```text
5660a82 Add guarded workflow lifecycle API
```

This phase adds a LangGraph lifecycle summary, explicit `awaiting_data` status, state-transition guards, persistent workflow events, the `GET /projects/{id}/workflow` endpoint, frontend handoff updates, and lifecycle tests.

Verification command:

```bash
..\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
```

Result: `5 passed`. Two non-failing warnings remain: a Starlette/AnyIO deprecation warning and a joblib Windows CPU-detection warning.

## Run-planning phase record

Implemented on `codex/run-planning`. This phase adds the deterministic run-plan service and `GET /projects/{id}/run-plan`, documenting the candidate estimators, validation strategy, metric direction, and reproducibility settings before training begins.

Verification remained `5 passed` with the same two non-failing warnings.

## Data-readiness hardening phase record

Implemented on `codex/data-readiness-hardening`. This phase expands upload profiling into a pre-flight gate that blocks invalid regression targets, insufficient class support, and unusable features. It also separates actionable warnings from run-blocking errors for the frontend.

Verification result: `6 passed` with the same two non-failing environment warnings.

## API operational configuration phase record

Implemented on `codex/api-operational-config`. This phase adds explicit browser-origin allow-list configuration, production API-key safeguards, `/health` and `/ready` operational endpoints, a reference environment template, and frontend integration guidance.

Verification result: `7 passed` with the same two non-failing environment warnings. The asynchronous-run polling in tests was increased from five to ten seconds to remove an observed Windows timing flake.

## Model report and explainability phase record

Implemented on `codex/model-report-explainability`. This phase persists a model card with validation settings, data summary, documented limitations, and permutation-based feature impact. The impact is explicitly labelled exploratory rather than causal.

Verification result: `7 passed` with the same two non-failing environment warnings.

## Durable job execution and observability phase record

Implemented on `codex/durable-job-observability`. This phase adds durable run events, atomic run-state/event commits, database-guarded single active runs, structured retryable failure messages, and startup reconciliation for jobs interrupted by the local server process.

Verification result: `7 passed` with the same two non-failing environment warnings.

## Model-inference phase record

Implemented on `codex/model-inference`. This phase adds guarded batch inference for completed models. It validates requested columns against the recorded training features and returns class probabilities when the fitted classifier supports them.

Verification result: `6 passed` with the same two non-failing environment warnings.

## Model-artifact phase record

Implemented on `codex/model-artifacts`. This phase adds authenticated completed-model download, safe server-side artifact resolution, and a project run-history endpoint for the results screen.

Verification result: `6 passed` with the same two non-failing environment warnings.
