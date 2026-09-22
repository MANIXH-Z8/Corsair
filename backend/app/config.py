from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
ENVIRONMENT = os.getenv("AUTOBUILD_ENVIRONMENT", "development").lower()
DATA_DIR = Path(os.getenv("AUTOBUILD_DATA_DIR", BASE_DIR / "data")).resolve()
DATABASE_PATH = DATA_DIR / "autobuild.db"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"
TEMPLATE_DIR = DATA_DIR / "templates"
API_KEY = os.getenv("AUTOBUILD_API_KEY", "local-development-key")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CORS_ORIGINS = tuple(
    origin.strip() for origin in os.getenv("AUTOBUILD_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if origin.strip()
)

if ENVIRONMENT not in {"development", "test", "production"}:
    raise RuntimeError("AUTOBUILD_ENVIRONMENT must be development, test, or production")
if ENVIRONMENT == "production" and API_KEY == "local-development-key":
    raise RuntimeError("AUTOBUILD_API_KEY must be changed before running in production")
if not CORS_ORIGINS:
    raise RuntimeError("Set at least one AUTOBUILD_CORS_ORIGINS value")

for directory in (DATA_DIR, UPLOAD_DIR, ARTIFACT_DIR, TEMPLATE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
