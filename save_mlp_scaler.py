"""
save_mlp_scaler.py — Extracts and saves the scaler used in MLP training
so api.py can apply it before predictions.

Run from project root:
    python save_mlp_scaler.py
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS

print("Fitting scaler on training data...")

# Load the same data used for training
df = pd.read_csv("data/students.csv")

# Apply the same feature engineering as api.py
df["study_attendance_ratio"] = df["study_hours_per_week"] * df["attendance_rate"]
df["academic_effort_score"]  = (df["study_hours_per_week"] * 0.4 +
                                 df["assignments_completed"] * 0.3 +
                                 df["tutoring_sessions"] * 0.3)
df["lifestyle_balance"]      = df["sleep_hours"] - df["part_time_job"] * 2

all_cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
X = df[all_cols].values

# Fit scaler on full dataset (same way sklearn Pipeline would)
scaler = StandardScaler()
scaler.fit(X)

# Save it
joblib.dump(scaler, "outputs/mlp_scaler.pkl")
print("Saved: outputs/mlp_scaler.pkl")

# Verify — test a prediction with and without scaler
mlp_clf = joblib.load("outputs/mlp_classifier.pkl")
mlp_reg = joblib.load("outputs/mlp_regressor.pkl")

test_row = np.array([[15, 0.85, 3.0, 85, 3, 7.0, 0, 2, 6,
                       15*0.85, 15*0.4+85*0.3+3*0.3, 7.0-0*2]])

raw_pred   = mlp_clf.predict(test_row)[0]
raw_score  = mlp_reg.predict(test_row)[0]

scaled     = scaler.transform(test_row)
scaled_pred  = mlp_clf.predict(scaled)[0]
scaled_score = mlp_reg.predict(scaled)[0]

print(f"\nWithout scaler → grade: {raw_pred}, score: {raw_score:.1f}")
print(f"With scaler    → grade: {scaled_pred}, score: {scaled_score:.1f}")
print(f"\nScore should be 0-100. Use scaler={'yes' if 0 <= scaled_score <= 100 else 'no — check preprocess.py'}")