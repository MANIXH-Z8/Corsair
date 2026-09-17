from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("AUTOBUILD_DATA_DIR", BASE_DIR / "data")).resolve()
DATABASE_PATH = DATA_DIR / "autobuild.db"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"
TEMPLATE_DIR = DATA_DIR / "templates"
API_KEY = os.getenv("AUTOBUILD_API_KEY", "local-development-key")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

for directory in (DATA_DIR, UPLOAD_DIR, ARTIFACT_DIR, TEMPLATE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
