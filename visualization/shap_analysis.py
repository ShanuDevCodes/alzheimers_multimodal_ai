import shap
import matplotlib.pyplot as plt
import os
from utils.logging_utils import setup_logger

logger = setup_logger("SHAPAnalysis")

def analyze_tabular_shap(model, X_df, feature_names, save_path="results/shap_summary.png"):
    
    logger.info("Computing SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    
    plt.figure()
    shap.summary_plot(shap_values, X_df, feature_names=feature_names, show=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved SHAP summary plot to {save_path}")
    return shap_values
