import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

COLUMN_RENAME_MAP = {
    "PatientID":                "patient_id",
    "Age":                      "age",
    "Gender":                   "gender",
    "Ethnicity":                "ethnicity",
    "EducationLevel":           "education_level",
    "BMI":                      "bmi",
    "Smoking":                  "smoking",
    "AlcoholConsumption":       "alcohol_consumption",
    "PhysicalActivity":         "physical_activity",
    "DietQuality":              "diet_quality",
    "SleepQuality":             "sleep_quality",
    "FamilyHistoryAlzheimers":  "family_history_alzheimers",
    "CardiovascularDisease":    "cardiovascular_disease",
    "Diabetes":                 "diabetes",
    "Depression":               "depression",
    "HeadInjury":               "head_injury",
    "Hypertension":             "hypertension",
    "SystolicBP":               "systolic_bp",
    "DiastolicBP":              "diastolic_bp",
    "CholesterolTotal":         "cholesterol_total",
    "CholesterolLDL":           "cholesterol_ldl",
    "CholesterolHDL":           "cholesterol_hdl",
    "CholesterolTriglycerides": "cholesterol_triglycerides",
    "MMSE":                     "mmse_score",
    "FunctionalAssessment":     "functional_assessment",
    "MemoryComplaints":         "memory_complaints",
    "BehavioralProblems":       "behavioral_problems",
    "ADL":                      "adl_score",
    "Confusion":                "confusion",
    "Disorientation":           "disorientation",
    "PersonalityChanges":       "personality_changes",
    "DifficultyCompletingTasks":"difficulty_completing_tasks",
    "Forgetfulness":            "forgetfulness",
    "Diagnosis":                "Diagnosis",
}

DROP_COLS = ["DoctorInCharge"]

class TabularPreprocessor:
    
    def __init__(self, label_col: str = "Diagnosis"):
        self.label_col = label_col
        self.preprocessor = None
        self.feature_names_out = []

        self.categorical_cols = [
            "gender",
            "ethnicity",
            "education_level",
        ]

        self.binary_cols = [
            "smoking",
            "family_history_alzheimers",
            "cardiovascular_disease",
            "diabetes",
            "depression",
            "head_injury",
            "hypertension",
            "memory_complaints",
            "behavioral_problems",
            "confusion",
            "disorientation",
            "personality_changes",
            "difficulty_completing_tasks",
            "forgetfulness",
        ]

        self.numerical_cols = [
            "age",
            "bmi",
            "alcohol_consumption",
            "physical_activity",
            "diet_quality",
            "sleep_quality",
            "systolic_bp",
            "diastolic_bp",
            "cholesterol_total",
            "cholesterol_ldl",
            "cholesterol_hdl",
            "cholesterol_triglycerides",
            "mmse_score",
            "functional_assessment",
            "adl_score",
        ]

    def _rename_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.rename(columns=COLUMN_RENAME_MAP)
        drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")
        return df

    def fit_transform(self, df: pd.DataFrame):
        
        df = self._rename_and_clean(df)

        cat_cols  = [c for c in self.categorical_cols if c in df.columns]
        num_cols  = [c for c in (self.numerical_cols + self.binary_cols) if c in df.columns]
        all_feat  = num_cols + cat_cols

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )

        X = df[all_feat]
        X_proc = self.preprocessor.fit_transform(X)

        cat_feat_names = (
            self.preprocessor.named_transformers_["cat"]
            .get_feature_names_out(cat_cols)
            .tolist()
            if cat_cols else []
        )
        self.feature_names_out = num_cols + cat_feat_names

        df_out = pd.DataFrame(X_proc, columns=self.feature_names_out)
        df_out["patient_id"] = df["patient_id"].values if "patient_id" in df.columns else np.arange(len(df))
        if self.label_col in df.columns:
            df_out[self.label_col] = df[self.label_col].values

        return df_out, self.feature_names_out

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
        df = self._rename_and_clean(df)
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        num_cols = [c for c in (self.numerical_cols + self.binary_cols) if c in df.columns]
        all_feat = num_cols + cat_cols
        X_proc = self.preprocessor.transform(df[all_feat])
        df_out = pd.DataFrame(X_proc, columns=self.feature_names_out)
        df_out["patient_id"] = df["patient_id"].values if "patient_id" in df.columns else np.arange(len(df))
        if self.label_col in df.columns:
            df_out[self.label_col] = df[self.label_col].values
        return df_out, self.feature_names_out

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "preprocessor":      self.preprocessor,
            "feature_names_out": self.feature_names_out,
            "categorical_cols":  self.categorical_cols,
            "numerical_cols":    self.numerical_cols,
            "binary_cols":       self.binary_cols,
            "label_col":         self.label_col,
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.preprocessor      = data["preprocessor"]
        self.feature_names_out = data["feature_names_out"]
        self.categorical_cols  = data["categorical_cols"]
        self.numerical_cols    = data["numerical_cols"]
        self.binary_cols       = data.get("binary_cols", [])
        self.label_col         = data.get("label_col", "Diagnosis")
