"""
fix_retrain.py — Retrains the classifier with balanced classes and verified labels
Usage: python fix_retrain.py
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS

print("=" * 55)
print("RETRAINING WITH BALANCED CLASSES")
print("=" * 55)

# ── 1. Load dataset ───────────────────────────────────────
csv_path = Path("data/students.csv")
if not csv_path.exists():
    print("ERROR: data/students.csv not found.")
    print("Run: cd data && python generate_sample.py")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} records")
print(f"Grade distribution:\n{df['grade'].value_counts()}\n")

# ── 2. Feature engineering (matches api.py exactly) ──────
df["study_attendance_ratio"] = df["study_hours_per_week"] * df["attendance_rate"]
df["academic_effort_score"]  = (df["study_hours_per_week"] * 0.4 +
                                 df["assignments_completed"] * 0.3 +
                                 df["tutoring_sessions"] * 0.3)
df["lifestyle_balance"]      = df["sleep_hours"] - df["part_time_job"] * 2

all_cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
X = df[all_cols].values
y = df["grade"].values  # string labels: "Pass", "Average", "At-Risk"

print(f"Feature columns : {all_cols}")
print(f"Unique labels   : {np.unique(y)}")

# ── 3. Train/test split ───────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Compute class weights to handle imbalance ──────────
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"\nClass weights (balanced): {class_weight_dict}")

# ── 5. Train XGBoost with scale_pos_weight equivalent ─────
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"Label encoding  : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Compute sample weights from class weights
sample_weights = np.array([class_weight_dict[label] for label in y_train])

clf = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train_s, y_train_enc, sample_weight=sample_weights)

# ── 6. Evaluate ───────────────────────────────────────────
from sklearn.metrics import classification_report, accuracy_score

y_pred_enc = clf.predict(X_test_s)
y_pred     = le.inverse_transform(y_pred_enc)
y_test_dec = le.inverse_transform(y_test_enc)

print(f"\nTest Accuracy: {accuracy_score(y_test_dec, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_dec, y_pred))

# ── 7. Verify predictions make sense ─────────────────────
print("=" * 55)
print("SANITY CHECKS")
print("=" * 55)

def predict_label(features_raw):
    features_dict = dict(zip(FEATURE_COLS, features_raw[:9]))
    features_dict["study_attendance_ratio"] = features_raw[0] * features_raw[1]
    features_dict["academic_effort_score"]  = features_raw[0]*0.4 + features_raw[3]*0.3 + features_raw[4]*0.3
    features_dict["lifestyle_balance"]      = features_raw[5] - features_raw[6]*2
    X_in = np.array([[features_dict[c] for c in all_cols]])
    X_sc = scaler.transform(X_in)
    enc_pred = clf.predict(X_sc)[0]
    proba    = clf.predict_proba(X_sc)[0]
    label    = le.inverse_transform([enc_pred])[0]
    conf     = dict(zip(le.classes_, proba.round(3)))
    return label, conf

# study, attend, gpa, assign, tutor, sleep, job, extra, socio
strong = [22, 0.97, 3.9, 99, 6, 8.0, 0, 2, 9]
weak   = [4,  0.45, 1.2, 50, 0, 4.5, 1, 0, 2]
mid    = [14, 0.80, 2.8, 75, 2, 7.0, 0, 2, 6]

for name, feat in [("Strong (should=Pass)", strong),
                   ("Weak   (should=At-Risk)", weak),
                   ("Middle (should=Average)", mid)]:
    label, conf = predict_label(feat)
    print(f"{name} → {label}  | {conf}")

# ── 8. Save everything ────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)

# Save the XGBoost model — but store original string labels inside it
# by attaching the label encoder so api.py can decode properly
clf._label_encoder = le
joblib.dump(clf, "outputs/xgboost_classifier.pkl")
joblib.dump(scaler, "outputs/xgboost_scaler.pkl")
print("\n[SAVED] outputs/xgboost_classifier.pkl")
print("[SAVED] outputs/xgboost_scaler.pkl")

# ── 9. Print the correct LABEL_MAP for api.py ─────────────
print("\n" + "=" * 55)
print("COPY THIS INTO api.py  →  LABEL_MAP in run_prediction()")
print("=" * 55)
mapping = {str(i): label for i, label in enumerate(le.classes_)}
print(f"LABEL_MAP = {mapping}")
print("\nDone. Restart uvicorn and test again.")