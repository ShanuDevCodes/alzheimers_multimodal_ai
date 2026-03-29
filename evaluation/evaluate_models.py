import os
from utils.logging_utils import setup_logger
from evaluation.metrics import calculate_classification_metrics, plot_confusion_matrix

logger = setup_logger("EvaluateModels")

def evaluate_predictions(y_true, y_prob, model_name, output_dir="results"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    logger.info(f"--- Evaluation for {model_name} ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    return metrics

if __name__ == "__main__":
    logger.info("Evaluation module ready. Call evaluate_predictions from your test loops.")
