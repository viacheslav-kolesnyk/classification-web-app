import io
import os
import uuid
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify
)
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

import config
from classifier import WasteClassifier, CLASS_METADATA, CLASS_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["SAMPLE_FOLDER"] = config.SAMPLE_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Initialize the classifier instance (singleton for performance)
classifier = WasteClassifier(model_path=config.MODEL_PATH, device=config.DEVICE)


def allowed_file(filename: str) -> bool:
    """Checks whether the file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def validate_image(file_bytes: bytes):
    """Verifies that byte stream contains a valid image and reopens it for processing."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image.verify()
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # Примусове завантаження даних у пам'ять
        return img
    except (UnidentifiedImageError, OSError):
        return None


@app.route("/")
def index():
    """Page 1: Image Upload and Sample Selection."""
    samples = [
        {"name": "cardboard_sample.jpg", "label": "Cardboard Box", "category": "cardboard"},
        {"name": "glass_sample.jpg", "label": "Glass Bottle", "category": "glass"},
        {"name": "metal_sample.jpg", "label": "Aluminium Can", "category": "metal"},
        {"name": "paper_sample.jpg", "label": "Printed Paper", "category": "paper"},
        {"name": "plastic_sample.jpg", "label": "Plastic Bottle", "category": "plastic"},
        {"name": "trash_sample.jpg", "label": "Organic Residual", "category": "trash"},
    ]
    return render_template("index.html", samples=samples, class_names=CLASS_NAMES, metadata=CLASS_METADATA)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload or sample selection, runs inference, and displays results."""
    sample_name = request.form.get("sample_image")

    # 1. Check if user selected one of the built-in sample images
    if sample_name:
        safe_sample_name = secure_filename(sample_name)
        sample_path = Path(app.config["SAMPLE_FOLDER"]) / safe_sample_name
        if safe_sample_name and sample_path.is_file():
            try:
                results = classifier.predict(sample_path)
                image_url = url_for("static", filename=f"samples/{safe_sample_name}")
                return render_template("result.html", result=results, image_url=image_url, is_sample=True)
            except Exception as e:
                logger.error(f"Error predicting sample image: {e}")
                flash(f"Error processing sample image: {str(e)}", "danger")
                return redirect(url_for("index"))

    # 2. Check if user uploaded a file
    if "file" not in request.files:
        flash("No file part in the request.", "warning")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected for upload. Please choose an image.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file format. Please upload a JPG, PNG, WEBP, or BMP image.", "danger")
        return redirect(url_for("index"))

    file_bytes = file.read()
    image = validate_image(file_bytes)
    if image is None:
        return render_template(
            "error.html",
            error_code=400,
            error_title="Invalid Image",
            error_message="The uploaded file is not a valid image."
        ), 400

    # Save to uploads folder with unique UUID name to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = app.config["UPLOAD_FOLDER"] / unique_filename

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Run inference
    results = classifier.predict(image)
    image_url = url_for("static", filename=f"uploads/{unique_filename}")

    return render_template("result.html", result=results, image_url=image_url, is_sample=False)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """REST API endpoint for programmatic classification."""
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "error": "No file uploaded in form-data field 'file'"
        }), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "error": "Invalid file format or empty filename"
        }), 400

    file_bytes = file.read()
    image = validate_image(file_bytes)
    if image is None:
        return jsonify({
            "status": "error",
            "error": "The uploaded file is not a valid image."
        }), 400

    try:
        results = classifier.predict(image)
        return jsonify({
            "status": "success",
            "prediction": results
        }), 200
    except Exception as e:
        logger.error(f"API classification error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/about")
def about():
    """About Page: Project details, Model Architecture, Dataset, Metrics."""
    return render_template("about.html", metadata=CLASS_METADATA)


@app.errorhandler(413)
def request_entity_too_large(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "File size exceeds the 16MB limit. Please upload a smaller image."
        }), 413
    return render_template(
        "error.html",
        error_code=413,
        error_title="File Size Limit Exceeded",
        error_message=(
            "The uploaded file exceeds the 16MB limit. "
            "Please upload a smaller image."
        ),
    ), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "The requested resource was not found on this server."
        }), 404
    return render_template(
        "error.html",
        error_code=404,
        error_title="Page Not Found",
        error_message="The page you are looking for does not exist or has been moved."
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "An internal server error occurred."
        }), 500
    return render_template(
        "error.html",
        error_code=500,
        error_title="Internal Server Error",
        error_message="An unexpected server error occurred. Please try again later."
    ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
