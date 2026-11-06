import io
import pytest
from PIL import Image
from app import app
import config


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


def test_predict_with_uploaded_file(client):
    # Create a small valid test JPEG in memory
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
    assert b"No file part in the request" in response.data


def test_predict_invalid_extension(client):
    buf = io.BytesIO(b"not an image text")
    data = {
        "file": (buf, "document.pdf")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid file format" in response.data


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
