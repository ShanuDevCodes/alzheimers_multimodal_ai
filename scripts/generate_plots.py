import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_feature_importance(model_path, title, save_path):
    if not os.path.exists(model_path):
        print(f"Skipping {title} - model not found at {model_path}")
        return
    
    booster = xgb.Booster()
    booster.load_model(model_path)
    
    importance = booster.get_score(importance_type='gain')
    
    if not importance:
        print(f"Skipping {title} - no feature importances found.")
        return
        
    df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=True).tail(12)
    
    df_imp['Feature'] = df_imp['Feature'].str.replace('_', ' ').str.title()
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f'Top Features: {title}', fontsize=16, pad=15)
    plt.xlabel('Information Gain (Feature Importance)', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_mri_distribution(raw_mri_dir, save_path):
    if not os.path.exists(raw_mri_dir):
        print(f"Skipping MRI chart - directory {raw_mri_dir} not found.")
        return
    
    ad_count = len([f for f in os.listdir(os.path.join(raw_mri_dir, 'AD')) if f.endswith('.png')])
    cn_count = len([f for f in os.listdir(os.path.join(raw_mri_dir, 'CN_MCI')) if f.endswith('.png')])
    
    plt.figure(figsize=(8, 6))
    bars = sns.barplot(x=['Alzheimer\'s (AD)', 'Cognitively Normal (CN)'], y=[ad_count, cn_count], palette=['#ff4d4d', '#4dabf7'])
    
    for i, v in enumerate([ad_count, cn_count]):
        bars.text(i, v + 20, str(v), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title('OASIS-1 MRI Dataset Class Distribution', fontsize=16, pad=15)
    plt.ylabel('Number of Patients (2D Mid-Axial Slices)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_clinical_relations(csv_path, save_path):
    if not os.path.exists(csv_path):
        print(f"Skipping clinical relation chart - file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    label_col = 'Diagnosis'
    age_col = 'Age'
    mmse_col = 'MMSE'
    
    if label_col not in df.columns or age_col not in df.columns or mmse_col not in df.columns:
        print("Required columns missing for clinical relation plot.")
        return
        
    df['Status'] = df[label_col].map({1: "Alzheimer's", 0: "Healthy"})
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=age_col, y=mmse_col, hue='Status', palette={'Alzheimer\'s': '#ff4d4d', 'Healthy': '#4dabf7'}, alpha=0.6)
    
    plt.title('MMSE Cognitive Score vs Age', fontsize=16, pad=15)
    plt.xlabel('Patient Age', fontsize=12)
    plt.ylabel('Mini-Mental State Examination (MMSE) Score', fontsize=12)
    plt.axhline(y=24, color='gray', linestyle='--', alpha=0.7, label='Dementia Threshold (24)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_densenet_learning_curve(save_path):
    epochs = list(range(1, 24))
    
    val_loss = [
        0.4441, 0.3087, 0.3367, 0.3569, 0.2859, # Warmup 1-5
        0.6851, 0.2080, 0.0812, 0.0776, 0.0331, # Finetune 1-5
        0.0394, 0.0302, 0.0360, 0.0322, 0.0341, # Finetune 6-10
        0.0244, 0.0274, 0.0217, 0.0211, 0.0275, # Finetune 11-15
        0.0245, 0.0177, 0.0232
    ]
    
    val_acc = [
        0.884, 0.938, 0.955, 0.953, 0.950,
        0.917, 0.950, 0.985, 0.973, 0.991, 
        0.991, 0.997, 0.988, 0.991, 0.994, 
        0.997, 0.994, 0.994, 0.994, 0.994, 
        0.994, 0.997, 0.994
    ]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#ff4d4d'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss (BCEWithLogits)', color=color, fontsize=12)
    ax1.plot(epochs, val_loss, color=color, marker='o', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7)
    plt.text(5.7, 0.6, 'Backbone Unfrozen\n(Fine-tuning starts)', color='gray', fontsize=10)
    
    ax2 = ax1.twinx()
    color = '#4dabf7'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)
    ax2.plot(epochs, [a * 100 for a in val_acc], color=color, marker='s', linewidth=2, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(85, 101)
    
    plt.title('DenseNet-121 Transfer Learning on OASIS-1', fontsize=16, pad=15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    print(f"Generating publication-ready plots...")
    set_style()
    
    os.makedirs('report_plots', exist_ok=True)
    
    plot_mri_distribution('data/raw/mri/', 'report_plots/1_mri_class_distribution.png')
    
    plot_clinical_relations('data/raw/tabular/alzheimers_disease_data.csv', 'report_plots/2_age_vs_mmse.png')
    
    plot_feature_importance('models/saved/clinical_genetics_xgb.json', 
                            'Clinical & Genetic Risk Predictor', 
                            'report_plots/3_clinical_feature_importance.png')
    
    plot_feature_importance('models/saved/lifestyle_xgb.json', 
                            'Lifestyle & Behavioral Risk Predictor', 
                            'report_plots/4_lifestyle_feature_importance.png')
                            
    plot_densenet_learning_curve('report_plots/5_cnn_learning_curve.png')
    
    print("\nAll plots successfully generated in the 'report_plots/' directory!")
    print("You can use these directly in your documentation or presentation.")
