# GitHub Contribution Split for 5 Team Members

Here is a revised, highly focused breakdown. As requested, you will handle pushing the core configuration, inference, evaluation, and documentation scripts. 

Your 5 team members will *only* be responsible for pushing the core machine learning backend: the **models**, **training loops**, **datasets**, and **preprocessing** logic.

### 1. Person A: Deep Learning Data Engineer (MRI)
**Role:** Responsible for building the PyTorch dataset objects and preprocessing pipelines that handle raw 3D/2D brain scans.
**Files to Add & Commit:**
- `datasets/mri_dataset.py`
- `preprocessing/preprocess_mri.py`

```bash
git add datasets/mri_dataset.py preprocessing/preprocess_mri.py
git commit -m "feat: implement MRI PyTorch dataset object and imaging transformations"
git push
```

---

### 2. Person B: Deep Learning Architect (MRI)
**Role:** Responsible for the neural network architecture and training the network on the MRI dataset.
**Files to Add & Commit:**
- `models/cnn_model.py`
- `training/train_cnn.py`

```bash
git add models/cnn_model.py training/train_cnn.py
git commit -m "feat: design 2D DenseNet model and MRI training loop"
git push
```

---

### 3. Person C: Clinical Data Engineer (Tabular)
**Role:** Responsible for building the datasets, cleaning, and preprocessing the raw clinical and lifestyle Excel spreadsheets.
**Files to Add & Commit:**
- `datasets/tabular_dataset.py`
- `preprocessing/preprocess_tabular.py`

```bash
git add datasets/tabular_dataset.py preprocessing/preprocess_tabular.py
git commit -m "feat: implement tabular processors to encode and scale lifestyle/clinical data"
git push
```

---

### 4. Person D: Machine Learning Architect (Tabular)
**Role:** Responsible for the XGBoost tree algorithms that score the non-imaging clinical data.
**Files to Add & Commit:**
- `models/tabular_model.py`
- `models/lifestyle_model.py`
- `training/train_tabular.py`

```bash
git add models/tabular_model.py models/lifestyle_model.py training/train_tabular.py
git commit -m "feat: implement XGBoost models and training routines for clinical and lifestyle variables"
git push
```

---

### 5. Person E: Multimodal Fusion Architect
**Role:** Responsible for designing and training the final Multilayer Perceptron that fuses the MRI and Tabular risk predictions.
**Files to Add & Commit:**
- `models/fusion_model.py`
- `training/train_fusion.py`

```bash
git add models/fusion_model.py training/train_fusion.py
git commit -m "feat: build and train the final multimodal fusion neural network"
git push
```

> **Your Role:** Once your team members have successfully pushed these core processing blocks, you can run `git add .` to safely add and push everything else (The `scripts/`, `inference/`, `visualization/`, `demo/`, `configs/`, and Markdown documents)!
