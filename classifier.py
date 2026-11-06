import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import io

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_METADATA = {
    "cardboard": {
        "display_name": "Cardboard Packaging",
        "bin_category": "Recycling Bin (Paper & Cardboard)",
        "bin_color": "#1e88e5",
        "icon": "📦",
        "description": "Corrugated boxes, cardboard packaging, shipping cartons, and paperboard cartons.",
        "recycling_tip": "Flatten boxes completely to save bin volume. Remove any plastic tape or styrofoam inserts before disposal.",
        "environmental_impact": "Recycling 1 ton of cardboard saves 9 cubic yards of landfill space and 46 gallons of oil."
    },
    "glass": {
        "display_name": "Glass Containers",
        "bin_category": "Glass Recycling Bin",
        "bin_color": "#2e7d32",
        "icon": "🍾",
        "description": "Beverage bottles, condiment jars, and non-treated clear, green, or amber glass containers.",
        "recycling_tip": "Rinse containers lightly to remove residue. Keep metal lids separate unless specified by local guidelines.",
        "environmental_impact": "Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity."
    },
    "metal": {
        "display_name": "Metal & Aluminium Cans",
        "bin_category": "Recycling Bin (Metals & Plastics)",
        "bin_color": "#fbc02d",
        "icon": "🥫",
        "description": "Aluminium beverage cans, tin food cans, clean aluminum foil, and metal lids.",
        "recycling_tip": "Empty and lightly rinse cans. Crushing aluminium cans optimizes storage space in recycling bins.",
        "environmental_impact": "Recycling aluminium saves 95% of the energy needed to produce new aluminium from raw bauxite ore."
    },
    "paper": {
        "display_name": "Clean Paper & Documents",
        "bin_category": "Paper Recycling Bin",
        "bin_color": "#0288d1",
        "icon": "📄",
        "description": "Office paper, newspapers, magazines, envelopes, notebooks, and clean printed materials.",
        "recycling_tip": "Ensure paper is dry and free of greasy food stains. Shredded paper should be bagged per local guidelines.",
        "environmental_impact": "Recycling 1 ton of paper saves 17 trees, 7,000 gallons of water, and 4,000 kilowatt-hours of electricity."
    },
    "plastic": {
        "display_name": "Plastic Bottles & Containers",
        "bin_category": "Plastics Recycling Bin",
        "bin_color": "#fb8c00",
        "icon": "🧴",
        "description": "PET beverage bottles, HDPE detergent bottles, food tubs, and rigid plastic containers.",
        "recycling_tip": "Empty all liquids, rinse food containers, and reattach caps to bottles before recycling.",
        "environmental_impact": "Recycling plastics saves crude oil and significantly reduces ocean microplastic pollution."
    },
    "trash": {
        "display_name": "General Residual Waste",
        "bin_category": "General Waste (Non-Recyclable / Landfill)",
        "bin_color": "#546e7a",
        "icon": "🗑️",
        "description": "Non-recyclable composite items, contaminated packaging, food scraps, and residual domestic waste.",
        "recycling_tip": "Dispose in general waste bin. Consider composting organic matter to minimize landfill contribution.",
        "environmental_impact": "Proper segregation ensures non-recyclables do not contaminate and ruin entire batches of recyclable materials."
    }
}


def build_architecture(num_classes: int = 6) -> nn.Module:
    """Builds a lightweight, high-performance MobileNetV3-Small classifier."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


class WasteClassifier:
    """Classifies waste images into 6 categories with confidence scores and disposal guidance."""

    def __init__(self, model_path: Union[str, Path, None] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_architecture(num_classes=len(CLASS_NAMES))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded trained model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {model_path}: {e}. Initializing default weights.")
        else:
            logger.info("Using baseline initialized model architecture.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> torch.Tensor:
        """Preprocesses an input image into a normalized tensor ready for model inference."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert to RGB mode (handles PNG alpha channel or grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        return tensor.to(self.device)

    def predict(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Dict[str, Any]:
        """Runs inference and returns detailed predictions, probabilities, and guidance."""
        tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

        probs_list = probabilities.cpu().numpy().tolist()
        predicted_idx = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs_list[predicted_idx]) * 100.0

        # Build sorted list of all class predictions
        all_predictions = []
        for idx, (c_name, prob) in enumerate(zip(CLASS_NAMES, probs_list)):
            meta = CLASS_METADATA[c_name]
            all_predictions.append({
                "class": c_name,
                "display_name": meta["display_name"],
                "icon": meta["icon"],
                "probability": round(float(prob) * 100.0, 2),
                "bin_category": meta["bin_category"],
                "bin_color": meta["bin_color"]
            })

        # Sort descending by probability
        all_predictions.sort(key=lambda x: x["probability"], reverse=True)

        meta = CLASS_METADATA[predicted_class]
        return {
            "predicted_class": predicted_class,
            "display_name": meta["display_name"],
            "icon": meta["icon"],
            "confidence": round(confidence, 2),
            "bin_category": meta["bin_category"],
            "bin_color": meta["bin_color"],
            "description": meta["description"],
            "recycling_tip": meta["recycling_tip"],
            "environmental_impact": meta["environmental_impact"],
            "all_predictions": all_predictions,
            "top_3": all_predictions[:3]
        }
