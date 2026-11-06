import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Web App Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "ecosort-secret-key-2026")
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
SAMPLE_FOLDER = BASE_DIR / "static" / "samples"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Model Configuration
MODEL_PATH = BASE_DIR / "model" / "waste_classifier.pth"
DEVICE = "cpu"  # Fast and reliable for inference on server/local machine

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLE_FOLDER.mkdir(parents=True, exist_ok=True)
