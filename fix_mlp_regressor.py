"""
fix_mlp_regressor.py — Retrains MLP regressor with the correct scaler
Run from project root: python fix_mlp_regressor.py
"""
import sys, os, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS
from src.models import get_model

print("Retraining MLP regressor...")

df = pd.read_csv("data/students.csv")

# Feature engineering
df["study_attendance_ratio"] = df["study_hours_per_week"] * df["attendance_rate"]
df["academic_effort_score"]  = (df["study_hours_per_week"] * 0.4 +
                                 df["assignments_completed"] * 0.3 +
                                 df["tutoring_sessions"] * 0.3)
df["lifestyle_balance"]      = df["sleep_hours"] - df["part_time_job"] * 2

all_cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
X = df[all_cols].values
y_reg = df["final_score"].values  # ← numeric scores, not grade strings

# Same split as classifier (random_state=42, stratify on grade)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Load the scaler saved by fix_mlp_final.py
scaler = joblib.load("outputs/mlp_scaler.pkl")
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train regressor
mlp_reg = get_model("mlp", "regression")
mlp_reg.fit(X_train_s, y_train)

# Verify score range
preds = mlp_reg.predict(X_test_s)
print(f"Score range on test set: {preds.min():.1f} – {preds.max():.1f}  (should be ~0–100)")

# Quick sanity check
default = np.array([[15, 0.85, 3.0, 85, 3, 7.0, 0, 2, 6,
                     15*0.85, 15*0.4+85*0.3+3*0.3, 7.0-0*2]])
score = mlp_reg.predict(scaler.transform(default))[0]
print(f"Default student predicted score: {score:.1f}")

joblib.dump(mlp_reg, "outputs/mlp_regressor.pkl")
print("[SAVED] outputs/mlp_regressor.pkl")
print("Done — uvicorn will auto-reload.")