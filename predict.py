"""
predict.py — Standalone CLI inference script

Usage:
    python predict.py --model xgboost
    python predict.py --model mlp
    python predict.py  (prompts for input interactively)
"""

import argparse
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS

MODEL_DIR = "outputs/"

FEATURE_QUESTIONS = {
    "study_hours_per_week":   ("Study hours per week",          1.0,  40.0),
    "attendance_rate":        ("Attendance rate (0.0 - 1.0)",   0.0,   1.0),
    "previous_gpa":           ("Previous GPA (0.0 - 4.0)",      0.0,   4.0),
    "assignments_completed":  ("Assignments completed (%)",     50.0, 100.0),
    "tutoring_sessions":      ("Tutoring sessions attended",     0.0,  20.0),
    "sleep_hours":            ("Average sleep hours per night",  4.0,  10.0),
    "part_time_job":          ("Has part-time job? (0=No, 1=Yes)", 0,   1),
    "extracurriculars":       ("Number of extracurriculars",     0.0,   8.0),
    "socioeconomic_index":    ("Socioeconomic index (1 - 10)",   1.0,  10.0),
}


def engineer_features(features: dict) -> np.ndarray:
    study    = features["study_hours_per_week"]
    attend   = features["attendance_rate"]
    assign   = features["assignments_completed"]
    tutor    = features["tutoring_sessions"]
    sleep    = features["sleep_hours"]
    job      = features["part_time_job"]

    features["study_attendance_ratio"] = study * attend
    features["academic_effort_score"]  = study * 0.4 + assign * 0.3 + tutor * 0.3
    features["lifestyle_balance"]      = sleep - job * 2

    cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
    return np.array([[features[c] for c in cols]])


def load_artifacts(model_name: str):
    clf_path = os.path.join(MODEL_DIR, f"{model_name}_classifier.pkl")
    reg_path = os.path.join(MODEL_DIR, f"{model_name}_regressor.pkl")

    # Prefer tuned models if available
    tuned_clf = os.path.join(MODEL_DIR, f"{model_name}_tuned_classifier.pkl")
    if os.path.exists(tuned_clf):
        clf_path = tuned_clf
        print(f"[INFO] Using tuned classifier: {tuned_clf}")

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"No classifier found at {clf_path}. Run main.py first.")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"No regressor found at {reg_path}. Run main.py first.")

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg


def get_input_interactive() -> dict:
    print("\n── Student Input ─────────────────────────────")
    features = {}
    for key, (label, lo, hi) in FEATURE_QUESTIONS.items():
        while True:
            try:
                val = float(input(f"  {label} [{lo}–{hi}]: "))
                if lo <= val <= hi:
                    features[key] = val
                    break
                print(f"    ⚠ Please enter a value between {lo} and {hi}")
            except ValueError:
                print("    ⚠ Please enter a number")
    return features


def predict(features: dict, model_name: str = "xgboost"):
    clf, reg = load_artifacts(model_name)
    X = engineer_features(features)

    grade     = clf.predict(X)[0]
    proba     = clf.predict_proba(X)[0]
    score     = reg.predict(X)[0]
    classes   = clf.classes_

    confidence = dict(zip(classes, (proba * 100).round(1)))

    print("\n── Prediction Results ────────────────────────")
    print(f"  Predicted Grade    : {grade}")
    print(f"  Predicted Score    : {score:.1f} / 100")
    print(f"  Confidence")
    for cls, pct in sorted(confidence.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 5)
        print(f"    {cls:<10} {pct:5.1f}%  {bar}")
    print("──────────────────────────────────────────────")

    return {"grade": grade, "score": round(float(score), 1), "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Predict student grade")
    parser.add_argument("--model", choices=["xgboost", "mlp"], default="xgboost")
    args = parser.parse_args()

    features = get_input_interactive()
    predict(features, args.model)


if __name__ == "__main__":
    main()
