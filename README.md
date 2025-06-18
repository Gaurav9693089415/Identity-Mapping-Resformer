
---

# Identity-Mapping ResFormer: A Computer-Aided Diagnosis Model for Pneumonia X-Ray Images

This project implements a deep learning model designed to classify chest X-rays into four categories:

* **COVID**
* **Viral Pneumonia**
* **Lung Opacity**
* **Normal**

Built from scratch using PyTorch, the model employs an innovative ResFormer's identity-mapping transformer architecture and is optimized for medical image classification.

---

## Project Structure

```
Identity-Mapping-ResFormer/
├── app.py                # Flask application (local deployment)
├── best_model.pth        # Trained & saved ResFormer weights
├── requirements.txt      # Python dependencies
├── test_environment.py   # Python version validator
├── notebook.ipynb        # Data prep & training script
├── uploads/              # Temporary storage for uploaded X-rays
├── templates/
│   └── index.html        # Web UI template
├── README.md             # Project documentation (this file)
└── LICENSE               # Project license
```

---

## Installation

1. **Clone the repo & navigate into the project folder**

```bash
git clone https://github.com/Gaurav9693089415/Identity-Mapping-Resformer
cd Identity-Mapping-Resformer
```

2. **Create & activate a virtual environment (optional but recommended)**

```bash
python3 -m venv venv
source venv/bin/activate      # For Mac/Linux
venv\Scripts\activate         # For Windows
```

3. **Install the dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify Python environment**

```bash
python test_environment.py
```

---

## Usage

### 1. Train the Model (`notebook.ipynb`)

* **Prepare**: Adjust `zip_path` & `dataset_path` to match your dataset locations.
* **Run**: Execute the notebook to:

  * Load and prepare the dataset
  * Define and train the `IdentityMappingResFormer`
  * Save the best model as `best_model.pth`
* **Outputs**:

  * Training loss curve
  * Validation accuracy plot
  * Confusion matrix heatmap

### 2. Deploy Locally using Flask

```bash
python app.py
```

* Open your browser and visit: `http://localhost:5000`
* Upload a chest X-ray image (`.png`, `.jpg`, etc.)
* View:

  * Predicted class (COVID / Viral Pneumonia / Lung Opacity / Normal)
  * Confidence score
  * Full probability breakdown

**Sample JSON Response:**

```json
{
  "success": true,
  "prediction": {
    "predicted_class": "COVID",
    "confidence": 0.92,
    "all_probabilities": {
      "COVID": 0.92,
      "Viral Pneumonia": 0.05,
      "Lung_Opacity": 0.02,
      "Normal": 0.01
    }
  },
  "image": "<base64-encoded PNG>"
}
```

---

## Dependencies

All required packages are listed in `requirements.txt`:

```
Flask==3.0.0
torch==2.4.1
torchvision==0.19.1
Pillow==10.4.0
numpy==1.26.4
albumentations==1.4.0
einops==0.8.0
Werkzeug==3.0.1
```

---

## Model Architecture

* SimAM: Spatial attention mechanism
* DWConv3x3: Depthwise separable convolution
* MCCRM: Multi-channel convolution + residual mapping
* EMPT: Convolution-aware transformer block
* IMTM: Identity-Mapping Transformer module for feature fusion
* Backbone + Auxiliary network: Hierarchical stages for feature extraction
* Final classification head: Average pooling → Fully connected → 4-class output

---

## Evaluation Metrics

Example evaluation results after training:

* Accuracy: \~88.6%
* Precision: \~0.90
* Recall: \~0.89
* F1-score: \~0.895
* Specificity: \~0.953
* Confusion Matrix: Visualized in the notebook using seaborn heatmap

---

## Development & Testing

* Ensure Python 3.x is used (`test_environment.py` validates this)
* Runs on CPU (no GPU required)
* Uses `albumentations` for augmentation and `einops` for tensor reshaping
* Flask app handles:

  * Image validation
  * Preprocessing
  * Prediction with confidence
  * Base64 encoding for web display

---

## License

This project is released under the **MIT License**.

---


