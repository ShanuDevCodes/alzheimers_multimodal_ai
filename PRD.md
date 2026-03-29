# Product Requirements Document (PRD): Alzheimer's Multimodal AI Prediction System

## 1. Project Overview
The Alzheimer's Multimodal AI Prediction System is a production-quality, modular machine learning project designed to predict early Alzheimer's disease risk. The system achieves a highly accurate diagnosis by fusing disparate types of medical data—clinical records, lifestyle habits, genetic information, and brain imaging. This comprehensive evaluation better mimics the holistic approach of a medical professional diagnosing neurogenerative diseases.

## 2. Models and Algorithms Used
The project avoids a "one-size-fits-all" approach. Instead, it employs an ensemble / multimodal fusion architecture, utilizing distinct algorithms that excel at handling specific data types:

### A. MRI Image Processing (Deep Learning)
- **Algorithm/Model:** 2D DenseNet-121 (Convolutional Neural Network).
- **Purpose:** Serves as a feature extractor to analyze Brain MRI scans, producing 256-dimensional embeddings.
- **Why DenseNet?** Dense connections ensure deep feature propagation and help mitigate the vanishing-gradient problem. It is highly optimized for extracting texture and structural volume reductions (such as hippocampal shrinkage) inherent in Alzheimer's brains.

### B. Tabular Data Processing (Tree-Based Machine Learning)
- **Algorithm/Model:** XGBoost (Extreme Gradient Boosting), using an ensemble of sequential decision trees.
- **Structure:** Features two distinct sub-models:
  1. **Clinical & Genetics Model:** Evaluates non-modifiable risk factors and professional medical tests (e.g., age, APOE genotype, MMSE score, CDR score).
  2. **Lifestyle Model:** Evaluates modifiable everyday habits (e.g., diet score, physical activity, smoking, sleep).
- **Why XGBoost?** Excels at tabular data, inherently handles missing values well, and provides robust integration with SHAP out-of-the-box for explainability.

### C. Multimodal Fusion Model
- **Algorithm/Model:** Multilayer Perceptron (MLP) Neural Network.
- **Purpose:** Acts as the final adjudicator. It concatenates the 256D continuous MRI embeddings from the DenseNet with the continuous output risk scores from the two XGBoost models, rendering a final binary classification (0: Low Risk, 1: High Risk).

## 3. Training Data
The models are trained using two fundamentally differing sets of input data mapping to the same patient IDs:

### A. Brain MRI Images (3D / 2D Volumes)
- **Format:** Compressed NIfTI format (`.nii.gz`) converted to internal image processing matrices.
- **Content:** Structural T1-weighted brain MRI scans.
- **Sources:** Professional medical research datasets such as OASIS-1/3 (Open Access Series of Imaging Studies) and ADNI (Alzheimer's Disease Neuroimaging Initiative).

### B. Tabular Patient Records
- **Format:** Spreadsheets (`.xlsx` or `.csv`).
- **Required Columns & Features:**
  - **Identity:** `patient_id` (Unique Identifier)
  - **Demographics:** `age`, `gender`, `education_level`
  - **Clinical:** `mmse_score` (Mini-Mental State Examination), `cdr_score` (Clinical Dementia Rating), `memory_test_score`
  - **Genetics:** `apoe_genotype`
  - **Lifestyle:** `sleep_hours`, `physical_activity`, `smoking_status`, `diet_score`
  - **Target Variable (Label):** `Diagnosis` or `alzheimer_risk` (0 = Control/MCI, 1 = Alzheimer's)

## 4. How the Model is Trained
The training pipeline executes an orchestrated three-phase approach designed for scalability and robustness against class imbalances.

### Phase 1: Preprocessing & Tabular Model Training
1. **Data Leakage Prevention:** Raw data is immediately split using an 80% train / 20% test ratio. The preprocessor (scaling and encoding) is fitted *only* on the training split.
2. **Feature Separation:** The tabular data is automatically segregated into "Lifestyle" features vs. "Clinical/Genetic" features.
3. **Training:** Both separate XGBoost classification models are trained. If available, the training is offloaded efficiently to the GPU via CUDA. 

### Phase 2: MRI Deep Learning Model Training
1. **Data Augmentation:** MRI datasets are highly augmented automatically using transformation techniques to prevent overfitting.
2. **Class Imbalance Strategy:** Datasets typically have far more healthy brains (CN) than Alzheimer's brains (AD). The system calculates a dynamic `pos_weight` applied to a Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`) function to penalize misclassifying sick brains more heavily.
3. **Two-Step Fine-Tuning:**
   - **Warm-up:** The pre-trained DenseNet backbone is frozen. The model quickly trains only the newly attached classification head.
   - **Fine-Tuning:** The entire backbone is unfrozen and trained end-to-end at a much lower learning rate managed by a Cosine Annealing learning rate scheduler (`CosineAnnealingLR`).

### Phase 3: Multimodal Fusion Training 
1. **Feature Aggregation:** The finalized, frozen sub-models generate standardized "scores" and "embeddings" for each patient case.
2. **Final Aggregation Layer:** The Fusion PyTorch dataset passes these combined values into the Multilayer Perceptron.
3. **Optimization:** Standard backpropagation is performed on this final neural network layer using an Adam optimizer over several epochs until the loss converges. 

## 5. System Evaluation & Explainability for the Professor
To guarantee academic integrity and usability, the pipeline features:
- **Held-Out Demonstration Set:** A dedicated 20% slice of purely unseen MRIs and Tabular records are completely held out in a separate test directory. These will be used during the live demonstration to prove the model's predictive capacities isn't simply memorizing the training dataset.
- **Explainable AI (XAI):** 
  - **3D Grad-CAM:** Used against the MRI to generate visual heatmaps showing the professor precisely which anatomical regions of the brain the CNN looked at to determine the classification.
  - **SHAP Values:** Generated for the XGBoost models to mathematically rank which clinical and lifestyle factors contributed to the diagnosis.
