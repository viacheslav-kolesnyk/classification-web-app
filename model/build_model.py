"""
Model Weight Generator for EcoSort Waste Classifier.
Builds the MobileNetV3-Small architecture with pre-trained feature backbone
and saves the checkpoint for instant inference in the Flask web application.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models


def build_and_save_model(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load MobileNetV3-Small with default pre-trained weights
    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Replace final linear layer for 6 classes
    num_classes = 6
    in_features = base_model.classifier[3].in_features
    base_model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    # Save the model state dictionary
    torch.save(base_model.state_dict(), output_path)
    print(f"Model successfully saved to {output_path} (size: {output_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "waste_classifier.pth"
    build_and_save_model(out_file)
