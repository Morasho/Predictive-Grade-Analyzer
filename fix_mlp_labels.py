"""
fix_mlp_final.py — Fixes two issues:
1. MLP scaler was fit on full dataset (leakage) — refit on train split only
2. Aligns train/test split with XGBoost so both models are comparable

Run from project root:
    python fix_mlp_final.py
"""
import sys, os, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS
from src.models import get_model

print("=" * 55)
print("FIXING MLP SCALER + RETRAINING")
print("=" * 55)

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv("data/students.csv")

df["study_attendance_ratio"] = df["study_hours_per_week"] * df["attendance_rate"]
df["academic_effort_score"]  = (df["study_hours_per_week"] * 0.4 +
                                 df["assignments_completed"] * 0.3 +
                                 df["tutoring_sessions"] * 0.3)
df["lifestyle_balance"]      = df["sleep_hours"] - df["part_time_job"] * 2

all_cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
X = df[all_cols].values
y = df["grade"].values

# ── 2. Same split as XGBoost (same random_state=42) ──────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Fit scaler on TRAIN only (no leakage) ─────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit only on train
X_test_s  = scaler.transform(X_test)        # transform test

# ── 4. Encode labels ──────────────────────────────────────
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 5. Train MLP ──────────────────────────────────────────
mlp = get_model("mlp", "classification")
mlp.fit(X_train_s, y_train_enc)

# ── 6. Evaluate ───────────────────────────────────────────
y_pred = le.inverse_transform(mlp.predict(X_test_s))
y_true = le.inverse_transform(y_test_enc)
print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred))

# ── 7. Sanity checks ──────────────────────────────────────
print("=" * 55)
print("SANITY CHECKS (same student as index.html default)")
print("=" * 55)

def predict(raw_features):
    d = dict(zip(FEATURE_COLS, raw_features[:9]))
    d["study_attendance_ratio"] = raw_features[0] * raw_features[1]
    d["academic_effort_score"]  = raw_features[0]*0.4 + raw_features[3]*0.3 + raw_features[4]*0.3
    d["lifestyle_balance"]      = raw_features[5] - raw_features[6]*2
    X_in = np.array([[d[c] for c in all_cols]])
    X_sc = scaler.transform(X_in)
    enc  = mlp.predict(X_sc)[0]
    prob = mlp.predict_proba(X_sc)[0]
    return le.inverse_transform([enc])[0], dict(zip(le.classes_, prob.round(3)))

# study, attend, gpa, assign, tutor, sleep, job, extra, socio
default = [15, 0.85, 3.0, 85, 3, 7.0, 0, 2, 6]   # default form values
strong  = [22, 0.97, 3.9, 99, 6, 8.0, 0, 2, 9]
weak    = [4,  0.45, 1.2, 50, 0, 4.5, 1, 0, 2]

for name, feat in [("Default form values", default),
                   ("Strong  (should=Pass)", strong),
                   ("Weak    (should=At-Risk)", weak)]:
    label, conf = predict(feat)
    print(f"{name:30s} → {label:10s} | {conf}")

# ── 8. Also retrain MLP regressor with same scaler ────────
print("\nRetraining MLP regressor with same scaler...")
mlp_reg = get_model("mlp", "regression")
mlp_reg.fit(X_train_s, y_train[:len(X_train_s)])  # use raw scores

# Get regression target (final_score)
y_reg_train = df.loc[df.index[:len(X_train)], "final_score"].values
y_reg_test  = df.loc[df.index[len(X_train):], "final_score"].values

mlp_reg2 = get_model("mlp", "regression")
mlp_reg2.fit(X_train_s, y_reg_train[:len(X_train_s)])
preds = mlp_reg2.predict(X_test_s)
print(f"Score range on test set: {preds.min():.1f} – {preds.max():.1f}  (should be ~0–100)")

# ── 9. Save all ───────────────────────────────────────────
joblib.dump(mlp,     "outputs/mlp_classifier.pkl")
joblib.dump(mlp_reg2,"outputs/mlp_regressor.pkl")
joblib.dump(scaler,  "outputs/mlp_scaler.pkl")

print("\n[SAVED] mlp_classifier.pkl")
print("[SAVED] mlp_regressor.pkl")
print("[SAVED] mlp_scaler.pkl")
print("\nRestart uvicorn and both models should give consistent results.")