import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = os.environ.get("SECRET_KEY", "ecosort-secret-key-2026")
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
SAMPLE_FOLDER = BASE_DIR / "static" / "samples"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

MODEL_PATH = BASE_DIR / "model" / "waste_classifier.pth"
DEVICE = "cpu"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLE_FOLDER.mkdir(parents=True, exist_ok=True)
