import os
import argparse
import shutil
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.preprocess_tabular import TabularPreprocessor
from utils.logging_utils import setup_logger
from utils.config_loader import load_config

logger = setup_logger("PreprocessData")

def preprocess_all(config_path: str = "configs/default_config.yaml"):
    config       = load_config(config_path)
    tabular_path = config["paths"]["tabular_data"]
    processed_dir = config["paths"]["processed_dir"]
    model_save_dir = config["paths"]["model_save_dir"]
    mri_data_dir  = config["paths"]["mri_data"]
    test_dir      = config["paths"]["test_dir"]
    label_col     = config["training"].get("label_col", "Diagnosis")
    test_split    = float(config["training"].get("test_split", 0.20))
    seed          = int(config["training"].get("seed", 42))

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if not os.path.exists(tabular_path):
        logger.error(f"Tabular data not found at '{tabular_path}'. Aborting.")
        return

    logger.info(f"Reading tabular data from: {tabular_path}")
    if tabular_path.endswith(".xlsx") or tabular_path.endswith(".xls"):
        df_raw = pd.read_excel(tabular_path)
    else:
        df_raw = pd.read_csv(tabular_path)

    logger.info(f"  → {len(df_raw)} rows  |  {len(df_raw.columns)} columns")

    for drop_col in ["DoctorInCharge"]:
        if drop_col in df_raw.columns:
            df_raw = df_raw.drop(columns=[drop_col])

    df_train_raw, df_test_raw = train_test_split(
        df_raw,
        test_size=test_split,
        random_state=seed,
        stratify=df_raw[label_col] if label_col in df_raw.columns else None,
    )

    logger.info(
        f"  Train rows: {len(df_train_raw)}  |  Test rows: {len(df_test_raw)}  "
        f"(split {int((1-test_split)*100)}/{int(test_split*100)})"
    )

    preprocessor = TabularPreprocessor(label_col=label_col)

    df_train_proc, feature_names = preprocessor.fit_transform(df_train_raw)
    df_test_proc, _              = preprocessor.transform(df_test_raw)

    train_csv = os.path.join(processed_dir, "processed_tabular.csv")
    df_train_proc.to_csv(train_csv, index=False)
    logger.info(f"Saved processed TRAIN data → {train_csv}  ({len(feature_names)} features)")

    feat_txt = os.path.join(processed_dir, "tabular_features.txt")
    with open(feat_txt, "w") as f:
        f.write("\n".join(feature_names))

    prep_path = os.path.join(model_save_dir, "tabular_preprocessor.joblib")
    preprocessor.save(prep_path)
    logger.info(f"Saved fitted preprocessor → {prep_path}")

    test_tab_dir = os.path.join(test_dir, "tabular")
    os.makedirs(test_tab_dir, exist_ok=True)

    test_raw_csv = os.path.join(test_tab_dir, "test_data_raw.csv")
    df_test_raw.to_csv(test_raw_csv, index=False)

    test_proc_csv = os.path.join(test_tab_dir, "test_data_processed.csv")
    df_test_proc.to_csv(test_proc_csv, index=False)

    logger.info(f"Saved TEST tabular data (raw + processed) → {test_tab_dir}/")

    if label_col in df_test_proc.columns:
        dist = df_test_proc[label_col].value_counts().to_dict()
        logger.info(f"  Test set diagnosis distribution: {dist}")

    test_mri_dir = os.path.join(test_dir, "mri")
    os.makedirs(test_mri_dir, exist_ok=True)

    SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    all_mri = []
    if os.path.isdir(mri_data_dir):
        for ext in SUPPORTED_EXTS:
            all_mri.extend(glob.glob(os.path.join(mri_data_dir, "**", ext), recursive=True))
        all_mri = [f for f in all_mri if "_mask" not in os.path.basename(f).lower()]

    if all_mri:
        random.seed(seed)
        n_test_mri = min(max(10, int(len(all_mri) * test_split)), 100)
        test_mri_files = random.sample(all_mri, n_test_mri)

        copied = 0
        for src in test_mri_files:
            dst = os.path.join(test_mri_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

        logger.info(
            f"Copied {copied} MRI test images → {test_mri_dir}/  "
            f"(sampled {n_test_mri} from {len(all_mri)} total)"
        )
    else:
        logger.warning(
            f"No MRI images found in '{mri_data_dir}'. "
            "Run scripts/download_dataset.py first, then re-run preprocessing."
        )

    readme_path = os.path.join(test_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "ALZHEIMER'S MODEL — TEST DATA HOLDOUT\n"
            "======================================\n\n"
            "This directory contains data that was held out from training.\n"
            "Use it to demonstrate the model's performance to your professor.\n\n"
            "Contents:\n"
            "  tabular/test_data_raw.csv       — Original (human-readable) patient records\n"
            "  tabular/test_data_processed.csv — Scaled/encoded (model-ready) records\n"
            f"  mri/                            — {len(all_mri) if all_mri else 0} sample MRI brain images\n\n"
            f"Split ratio : {int((1-test_split)*100)}% train / {int(test_split*100)}% test\n"
            f"Random seed : {seed}\n"
            f"Label column: {label_col}  (0 = No Alzheimer's, 1 = Alzheimer's)\n"
        )

    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info(f"  Train data  → {processed_dir}")
    logger.info(f"  Test data   → {test_dir}  ← USE THIS FOR THE DEMO")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess all datasets.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    preprocess_all(args.config)
