# Alzheimer's Multimodal AI System

This project is a production-quality, modular, and reproducible machine learning system designed to predict early Alzheimer's disease risk using four distinct modalities:
1. Brain MRI images (3D volumes)
2. Genetic information (e.g., APOE genotype)
3. Lifestyle data
4. Clinical cognitive symptoms

## Features
- **Multimodal Fusion**: Combines Deep Learning for imaging with Gradient Boosting for tabular data.
- **Hardware Optimization**: Designed for training on GPU (CUDA) and optimized inference on CPU (TorchScript/ONNX).
- **Explainability**: Includes 3D Grad-CAM for MRI and SHAP for tabular features.

## Setup Instructions

### Environment Setup (WSL + CUDA)
Since training requires GPU acceleration on a desktop PC (RTX 4070 Super), it is recommended to run this inside WSL (Ubuntu) with CUDA enabled.

1. **Install WSL and Ubuntu**:
   Ensure WSL 2 is installed and running Ubuntu. You can install it via PowerShell: `wsl --install`.

2. **Install NVIDIA Drivers & CUDA Toolkit**:
   In Windows, ensure you have the latest NVIDIA drivers. Inside WSL, install the CUDA Toolkit compatible with your setup.

3. **Install Requirements**:
   Create a virtual environment and install the dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Dataset Preparation

The model requires two types of raw data placed in specific directories:

#### 1. Brain MRI Images (3D Volumes)
*   **Format**: `.nii.gz` (Compressed NIfTI)
*   **Location**: `data/raw/mri/`
*   **Content**: Structural T1-weighted scans.
*   **Sources**:
    *   [OASIS-3](https://www.oasis-brains.org/) (Open Access)
    *   [ADNI](https://adni.loni.usc.edu/) (Research gold standard)

#### 2. Tabular Data (Biomarkers & Lifestyle)
*   **Format**: `.csv`
*   **Filename**: `patient_data.csv`
*   **Location**: `data/raw/tabular/`
*   **Required Columns**:

| Category | Column Name | Description | Example |
| :--- | :--- | :--- | :--- |
| **Identity** | `patient_id` | Unique identifier | `sub_001` |
| **Demographics**| `age`, `gender`, `education_level` | Basic patient info | `72`, `M`, `Masters` |
| **Genetics** | `apoe_genotype` | Genetic risk factor | `e3/e4` |
| **Lifestyle** | `sleep_hours`, `physical_activity`, `smoking_status`, `diet_score` | Health habits | `7`, `High`, `Never`, `8` |
| **Clinical** | `mmse_score`, `cdr_score`, `memory_test_score` | Cognitive test results | `28`, `0.5`, `45.2` |
| **Target** | `alzheimer_risk` | **Label (0: Low, 1: High)** | `1` |

#### Data Preprocessing
Once you have placed the files, run the preprocessing pipeline:
```bash
python scripts/preprocess_data.py
```

## Training Pipeline
Train all individual modality models and the final fusion model by running:
```bash
python scripts/train_all.py
```

## Inference Pipeline
To run inference on CPU (e.g., on a laptop) using the exported optimized models:
```bash
python scripts/run_inference.py --mri path/to/scan.nii.gz --tabular path/to/patient_data.json
```
