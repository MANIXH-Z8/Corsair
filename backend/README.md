# AutoBuild backend MVP

API-first conversational AutoML for tabular classification and regression. The backend accepts a plain-language problem, requires an explicit ML-spec approval, profiles CSV/XLSX data, and trains real scikit-learn pipelines.

## Run

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:AUTOBUILD_API_KEY = "replace-this-before-sharing"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`; send `X-API-Key` with every endpoint other than `/health`.

## Frontend handoff

The frontend should render discovery, specification approval, data upload, data-readiness blockers, run progress, leaderboard, and report. Read the detailed contract in [docs/frontend-handoff.md](docs/frontend-handoff.md).

## MVP boundary

This is a private local MVP. It does not yet provide multi-user identities, Docker isolation, Redis jobs, PostgreSQL, live deployment, or external data connectors.
