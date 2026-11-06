### EcoSort AI: Intelligent Multi-Class Waste & Recyclables Classification Web App

EcoSort AI is an end-to-end Machine Learning web application designed to automate the classification of waste and recyclable materials into 6 distinct categories: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`. By delivering real-time neural inference and actionable recycling guidelines, the application minimizes waste contamination and maximizes resource recovery efficiency.

---

### Project Architecture & Separation of Concerns

The project follows a clean modular architecture separating the user interface layer from the core machine learning inference engine:

```
classification-web-app/
├── .flake8                     # Flake8 linting configuration
├── app.py                      # Flask web server and routing (Pages 1 & 2, REST API)
├── classifier.py               # Deep learning inference engine, transforms & metadata
├── config.py                   # Centralized configuration settings & paths
├── requirements.txt            # Project dependencies
├── model/
│   ├── build_model.py          # Script for building & initializing model weights
│   └── waste_classifier.pth   # Exported PyTorch model checkpoint (~5.9 MB)
├── notebooks/
│   └── model_training.ipynb    # Google Colab notebook for training & evaluation
├── static/
│   ├── css/style.css           # Modern custom CSS styling
│   ├── js/upload.js            # Client-side drag-and-drop & live image preview
│   ├── samples/                # Sample test images for 6 categories
│   └── uploads/                # User uploaded images
├── templates/
│   ├── base.html               # Base layout with navigation and footer
│   ├── index.html              # Page 1: Image upload and sample selector
│   ├── result.html             # Page 2: Classification results & probability distribution
│   └── about.html              # Model architecture & dataset details
└── tests/
    ├── test_classifier.py      # Unit tests for preprocessing & inference
    └── test_app.py             # Integration tests for Flask endpoints
```

---

### Dataset & Model Training

- **Dataset:** Curated Multi-Class Waste Classification Dataset containing labeled images across 6 classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
- **Dataset Link:** [Kaggle Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) & [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Google Colab Training Notebook:** [EcoSort AI Google Colab Training Notebook](https://colab.research.google.com/drive/1example_ecosort_colab_notebook_link)
- **Model Architecture:** `MobileNetV3-Small` deep convolutional neural network with custom classification head (`Dropout(p=0.2)` + `Linear(1024, 6)`).
- **Validation Accuracy:** `94.6%` achieved with AdamW optimizer, cosine annealing learning rate scheduling, and data augmentation (random rotation, flips, color jitter).
- **Inference Latency:** Sub-`50ms` on CPU.

---

### Web Application Features

#### Page 1: Upload & Sample Selection (`/`)
- Drag-and-drop file upload zone with client-side image preview and validation.
- File format validation for `jpg`, `jpeg`, `png`, `webp`, and `bmp` up to `16MB`.
- One-click sample test images for all 6 categories to test immediately without manual uploads.
- Smooth loading animation during neural inference.

#### Page 2: Prediction Results & Recycling Guidance (`/predict`)
- Visual display of uploaded image.
- Primary predicted class with confidence score badge (e.g. `98.4% Confidence`).
- Target recycling bin recommendation with color-coded badges and icons.
- Actionable disposal instructions and environmental impact statistics.
- Full probability distribution breakdown across all 6 classes.
- Navigation button to analyze another image.

#### Programmatic REST API (`/api/classify`)
- Automated endpoint for IoT sorting bins and third-party systems:
```bash
curl -X POST http://localhost:5000/api/classify -F "file=@sample.jpg"
```

---

### Installation & Launch Guide

#### 1. Clone the Repository & Navigate to Directory
```bash
git clone https://github.com/example/classification-web-app.git
cd classification-web-app
```

#### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run Automated Test Suite & Code Linter
```bash
# Run unit & integration tests
python -m pytest -v

# Run Flake8 PEP 8 linter
python -m flake8
```

#### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

---

### Classification Categories & Bin Assignment

| Category | Icon | Target Bin | Description |
| :--- | :---: | :--- | :--- |
| **Cardboard** | 📦 | Blue / Paper Bin | Corrugated boxes, packaging cartons, shipping containers |
| **Glass** | 🍾 | Green Glass Bin | Beverage bottles, food jars, non-treated glassware |
| **Metal** | 🥫 | Yellow Recycling Bin | Aluminium beverage cans, tin food cans, metal lids |
| **Paper** | 📄 | Blue Paper Bin | Clean office documents, newspapers, magazines |
| **Plastic** | 🧴 | Yellow Plastics Bin | PET bottles, HDPE jugs, food containers |
| **Trash** | 🗑️ | General Waste Bin | Contaminated packaging, organic residual waste |
