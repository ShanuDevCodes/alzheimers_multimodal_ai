import os
import shutil
import glob
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from PIL import Image
from utils.logging_utils import setup_logger
from utils.config_loader import load_config

logger = setup_logger("DatasetDownloader")

CDR_TO_BINARY = {0.0: 0, 0.5: 0, 1.0: 1, 2.0: 1}

def download_oasis1_data(output_mri_dir: str):
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        return None

    logger.info("Downloading OASIS-1 FastSurfer QuickSeg dataset from Kaggle (~27 GB)...")
    try:
        path = kagglehub.dataset_download(
            "mdfahimbinamin/oasis-1-fastsurfer-quickseg-segmentation-dataset"
        )
        logger.info(f"Dataset downloaded to: {path}")
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return None

    csv_candidates = glob.glob(os.path.join(path, "**", "final_oasis.csv"), recursive=True)
    if not csv_candidates:
        logger.error("Could not find final_oasis.csv")
        return path

    metadata_csv = csv_candidates[0]
    df_meta = pd.read_csv(metadata_csv)
    
    oasis_meta_out = "data/raw/tabular/oasis_metadata.csv"
    os.makedirs(os.path.dirname(oasis_meta_out), exist_ok=True)
    shutil.copy2(metadata_csv, oasis_meta_out)

    id_col  = 'ID'
    cdr_col = 'CDR_x'

    subject_label_map = {}
    if id_col in df_meta.columns and cdr_col in df_meta.columns:
        for _, row in df_meta.iterrows():
            subj_id = str(row[id_col]).strip()
            if not subj_id.endswith("_MR1"):
                subj_id += "_MR1"
            cdr_val = float(row[cdr_col]) if pd.notna(row[cdr_col]) else 0.0
            subject_label_map[subj_id] = CDR_TO_BINARY.get(cdr_val, 0)
    
    mgz_files = glob.glob(os.path.join(path, "**", "orig.mgz"), recursive=True)
    logger.info(f"Found {len(mgz_files)} 3D MRI volumes (orig.mgz).")

    ad_dir  = os.path.join(output_mri_dir, "AD")
    cn_dir  = os.path.join(output_mri_dir, "CN_MCI")
    os.makedirs(ad_dir, exist_ok=True)
    os.makedirs(cn_dir, exist_ok=True)

    copied_ad, copied_cn, unmatched = 0, 0, 0
    
    logger.info("Extracting middle 2D axial slice from each volume...")
    for src in mgz_files:
        parent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(src)))
        
        subj_id = "_".join(parent_dir_name.split("_")[:3])
        
        label = subject_label_map.get(subj_id, 0)
        if subj_id not in subject_label_map:
            unmatched += 1

        dst_dir = ad_dir if label == 1 else cn_dir
        dst_path = os.path.join(dst_dir, f"{parent_dir_name}.png")

        if not os.path.exists(dst_path):
            try:
                vol = nib.load(src).get_fdata() # standard: (W, H, Depth)
                
                depth = vol.shape[2]
                slice_2d = vol[:, :, depth // 2]
                
                slice_2d = np.clip(slice_2d, 0, np.percentile(slice_2d, 99)) # drop hot pixels
                if slice_2d.max() > 0:
                    slice_2d = slice_2d / slice_2d.max() * 255.0
                
                slice_2d = np.rot90(slice_2d)
                
                img = Image.fromarray(slice_2d.astype(np.uint8)).convert("L")
                img.save(dst_path)
                
                if label == 1: copied_ad += 1
                else: copied_cn += 1
            except Exception as e:
                logger.error(f"Failed to process {src}: {e}")

    logger.info(f"Extracted AD: {copied_ad} slices.")
    logger.info(f"Extracted CN_MCI: {copied_cn} slices.")
    if unmatched:
        logger.warning(f"{unmatched} unmapped IDs defaulted to CN_MCI.")

def convert_tabular_excel(xlsx_path: str, output_csv_path: str):
    if not os.path.exists(xlsx_path): return
    df = pd.read_excel(xlsx_path)
    if "DoctorInCharge" in df.columns: df = df.drop(columns=["DoctorInCharge"])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--skip_mri", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    xlsx_path = config["paths"]["tabular_data"]
    csv_out   = xlsx_path.replace(".xlsx", ".csv").replace(".xls", ".csv")
    convert_tabular_excel(xlsx_path, csv_out)

    if not args.skip_mri:
        download_oasis1_data(config["paths"]["mri_data"])
