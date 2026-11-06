import io
import pytest
from PIL import Image
import torch

from classifier import WasteClassifier, CLASS_NAMES
import config


@pytest.fixture
def classifier_instance():
    return WasteClassifier(model_path=config.MODEL_PATH, device="cpu")


def test_classifier_initialization(classifier_instance):
    assert classifier_instance is not None
    assert classifier_instance.model is not None
    assert isinstance(classifier_instance.model, torch.nn.Module)


def test_image_preprocessing_rgb(classifier_instance):
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    tensor = classifier_instance.preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)


def test_image_preprocessing_rgba_and_grayscale(classifier_instance):
    # RGBA format
    img_rgba = Image.new("RGBA", (150, 150), color=(100, 150, 200, 128))
    tensor_rgba = classifier_instance.preprocess_image(img_rgba)
    assert tensor_rgba.shape == (1, 3, 224, 224)

    # Grayscale 'L' format
    img_l = Image.new("L", (100, 100), color=128)
    tensor_l = classifier_instance.preprocess_image(img_l)
    assert tensor_l.shape == (1, 3, 224, 224)


def test_prediction_output_structure(classifier_instance):
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    result = classifier_instance.predict(img)

    assert "predicted_class" in result
    assert result["predicted_class"] in CLASS_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 100.0
    assert "display_name" in result
    assert "bin_category" in result
    assert "recycling_tip" in result
    assert "all_predictions" in result
    assert len(result["all_predictions"]) == 6

    # Verify probability distribution sum is approximately 100%
    prob_sum = sum(p["probability"] for p in result["all_predictions"])
    assert 99.0 <= prob_sum <= 101.0


def test_prediction_from_bytes(classifier_instance):
    img = Image.new("RGB", (100, 100), color=(50, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    result = classifier_instance.predict(img_bytes)
    assert result["predicted_class"] in CLASS_NAMES
