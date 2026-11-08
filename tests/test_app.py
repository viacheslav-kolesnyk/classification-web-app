import io
import pytest
from PIL import Image
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Intelligent Waste Classification" in response.data
    assert b"Upload Item Image" in response.data
    assert b"Try with Sample Test Images" in response.data


def test_about_route(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About EcoSort AI" in response.data
    assert b"MobileNetV3-Small" in response.data


def test_predict_with_sample_image(client):
    response = client.post("/predict", data={"sample_image": "cardboard_sample.jpg"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Classification Results" in response.data
    assert b"Class Probability Distribution" in response.data


def test_predict_with_invalid_or_malicious_sample_image(client):
    # Malicious or non-existent sample names should not match and should fall through to file upload check
    response = client.post("/predict", data={"sample_image": "../../../etc/passwd"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"No file selected for upload. Please choose an image." in response.data


def test_predict_with_uploaded_file(client):
    img = Image.new("RGB", (100, 100), color=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "file": (buf, "test_upload.jpg")
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Classification Results" in response.data
    assert b"Recycling Guideline" in response.data


def test_predict_without_file(client):
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"No file selected for upload. Please choose an image." in response.data


def test_predict_invalid_extension(client):
    buf = io.BytesIO(b"not an image text")
    data = {
        "file": (buf, "document.pdf")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid file format" in response.data


def test_predict_without_extension(client):
    buf = io.BytesIO(b"file without extension content")
    data = {
        "file": (buf, "noextensionfile")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid file format" in response.data


def test_predict_corrupted_image(client):
    # Pass arbitrary corrupted bytes with an allowed extension
    buf = io.BytesIO(b"corrupted image binary content")
    data = {
        "file": (buf, "corrupt.jpg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"The uploaded file is not a valid image." in response.data
    assert b"400" in response.data


def test_api_classify_endpoint(client):
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "file": (buf, "api_test.jpg")
    }

    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "prediction" in json_data
    assert "predicted_class" in json_data["prediction"]
    assert "confidence" in json_data["prediction"]


def test_api_classify_mocked_prediction(client, monkeypatch):
    import app as main_app

    mock_result = {
        "predicted_class": "plastic",
        "confidence": 0.985,
        "probabilities": {
            "plastic": 0.985,
            "glass": 0.010,
            "metal": 0.005,
            "paper": 0.0,
            "cardboard": 0.0,
            "trash": 0.0,
        },
        "description": "Plastic bottles, containers, bags",
        "bin": "Yellow / Recycling Bin",
        "icon": "bi-recycle",
    }

    monkeypatch.setattr(main_app.classifier, "predict", lambda img: mock_result)

    img = Image.new("RGB", (100, 100), color=(50, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {"file": (buf, "test_plastic.jpg")}
    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["prediction"]["predicted_class"] == "plastic"
    assert json_data["prediction"]["confidence"] == 0.985
    assert json_data["prediction"]["bin"] == "Yellow / Recycling Bin"
    assert json_data["prediction"]["probabilities"]["plastic"] == 0.985


def test_api_classify_rejects_corrupt_image(client):
    response = client.post(
        "/api/classify",
        data={
            "file": (
                io.BytesIO(b"this is not a real image"),
                "broken.jpg",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "valid image" in body["error"].lower()


def test_api_classify_rejects_unsupported_file_type(client):
    response = client.post(
        "/api/classify",
        data={
            "file": (
                io.BytesIO(b"some text content"),
                "payload.txt",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "invalid file format" in body["error"].lower()


def test_api_classify_corrupted_image(client):
    buf = io.BytesIO(b"corrupted image binary content")
    data = {
        "file": (buf, "corrupt.jpg")
    }
    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert json_data["error"] == "The uploaded file is not a valid image."


def test_api_classify_no_file(client):
    response = client.post("/api/classify", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "No file uploaded" in json_data["error"]


def test_api_classify_invalid_extension(client):
    buf = io.BytesIO(b"some pdf content")
    data = {
        "file": (buf, "document.pdf")
    }
    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "Invalid file format" in json_data["error"]


def test_api_classify_without_extension(client):
    buf = io.BytesIO(b"some content")
    data = {
        "file": (buf, "noextensionfile")
    }
    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "Invalid file format" in json_data["error"]


def test_404_error_html(client):
    response = client.get("/nonexistent-page-url")
    assert response.status_code == 404
    assert b"Page Not Found" in response.data
    assert b"404" in response.data
    assert b"Return to Classifier" in response.data


def test_404_error_api(client):
    response = client.get("/api/nonexistent-endpoint")
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "not found" in json_data["error"].lower()


def test_413_error_html(client):

    original_max = app.config.get("MAX_CONTENT_LENGTH")
    app.config["MAX_CONTENT_LENGTH"] = 100
    try:
        data = {
            "file": (io.BytesIO(b"A" * 500), "oversized.jpg")
        }
        response = client.post("/predict", data=data, content_type="multipart/form-data")
        assert response.status_code == 413
        assert b"File Size Limit Exceeded" in response.data
        assert b"413" in response.data
    finally:
        app.config["MAX_CONTENT_LENGTH"] = original_max


def test_413_error_api(client):
    original_max = app.config.get("MAX_CONTENT_LENGTH")
    app.config["MAX_CONTENT_LENGTH"] = 100
    try:
        data = {
            "file": (io.BytesIO(b"A" * 500), "oversized.jpg")
        }
        response = client.post("/api/classify", data=data, content_type="multipart/form-data")
        assert response.status_code == 413
        json_data = response.get_json()
        assert json_data["status"] == "error"
        assert "exceeds" in json_data["error"].lower()
    finally:
        app.config["MAX_CONTENT_LENGTH"] = original_max


def test_predict_sample_exception_handling(client, monkeypatch):
    import app as main_app

    def mock_predict(path_or_image):
        raise RuntimeError("Internal database connection failed: /etc/secrets.env")

    monkeypatch.setattr(main_app.classifier, "predict", mock_predict)
    response = client.post("/predict", data={"sample_image": "cardboard_sample.jpg"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Unable to process the sample image." in response.data
    assert b"/etc/secrets.env" not in response.data


def test_predict_upload_exception_handling(client, monkeypatch):
    import app as main_app
    from pathlib import Path

    uploads_before = set(Path(app.config["UPLOAD_FOLDER"]).glob("*"))

    def mock_predict(path_or_image):
        raise RuntimeError("Internal CUDA memory error: /models/weight.bin")

    monkeypatch.setattr(main_app.classifier, "predict", mock_predict)
    img = Image.new("RGB", (100, 100), color=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"file": (buf, "test_upload.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Unable to process the uploaded image." in response.data
    assert b"/models/weight.bin" not in response.data

    uploads_after = set(Path(app.config["UPLOAD_FOLDER"]).glob("*"))
    assert uploads_after == uploads_before


def test_predict_upload_cleanup_oserror_handling(client, monkeypatch):
    import app as main_app
    from pathlib import Path

    def mock_predict(path_or_image):
        raise RuntimeError("Inference failure")

    def mock_unlink(self):
        raise OSError("Permission denied")

    monkeypatch.setattr(main_app.classifier, "predict", mock_predict)
    monkeypatch.setattr(Path, "unlink", mock_unlink)

    img = Image.new("RGB", (100, 100), color=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"file": (buf, "test_upload_oserror.jpg")}
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Unable to process the uploaded image." in response.data


def test_api_classify_exception_handling(client, monkeypatch):
    import app as main_app

    def mock_predict(path_or_image):
        raise RuntimeError("Internal GPU kernel fault at 0xdeadbeef")

    monkeypatch.setattr(main_app.classifier, "predict", mock_predict)
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"file": (buf, "api_test.jpg")}
    response = client.post("/api/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 500
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert json_data["error"] == "An error occurred during classification."
    assert "0xdeadbeef" not in str(response.data)
