"""
tune_run.py — Run hyperparameter tuning and evaluate best models

Usage:
    python tune_run.py              # tune both XGBoost and MLP
    python tune_run.py --model xgboost
    python tune_run.py --model mlp
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.preprocess import prepare_data
from src.tune import tune_xgboost, tune_mlp
from src.evaluate import evaluate_classification, plot_feature_importance

DATA_PATH = "data/students.csv"
OUTPUT_DIR = "outputs/"

FEATURE_NAMES = [
    "study_hours_per_week", "attendance_rate", "previous_gpa",
    "assignments_completed", "tutoring_sessions", "sleep_hours",
    "part_time_job", "extracurriculars", "socioeconomic_index",
    "study_attendance_ratio", "academic_effort_score", "lifestyle_balance",
]


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--model", choices=["xgboost", "mlp", "both"], default="both")
    args = parser.parse_args()

    print("\n🔧 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le, _ = prepare_data(
        DATA_PATH, task="classification", oversample=False
    )
    class_names = list(le.classes_)

    results = {}

    if args.model in ("xgboost", "both"):
        best_xgb, xgb_params, xgb_cv = tune_xgboost(X_train, y_train, OUTPUT_DIR)
        print("\n📊 Evaluating tuned XGBoost on test set...")
        acc, _ = evaluate_classification(best_xgb, X_test, y_test, class_names, OUTPUT_DIR)
        plot_feature_importance(best_xgb, FEATURE_NAMES, OUTPUT_DIR, "xgboost_tuned")
        results["xgboost_tuned"] = {"cv": xgb_cv, "test": acc, "params": xgb_params}

    if args.model in ("mlp", "both"):
        best_mlp, mlp_params, mlp_cv = tune_mlp(X_train, y_train, OUTPUT_DIR)
        print("\n📊 Evaluating tuned MLP on test set...")
        acc, _ = evaluate_classification(best_mlp, X_test, y_test, class_names, OUTPUT_DIR)
        results["mlp_tuned"] = {"cv": mlp_cv, "test": acc, "params": mlp_params}

    print("\n" + "="*50)
    print("TUNING SUMMARY")
    print("="*50)
    for name, r in results.items():
        print(f"\n{name}")
        print(f"  CV Accuracy  : {r['cv']:.4f} ({r['cv']*100:.2f}%)")
        print(f"  Test Accuracy: {r['test']:.4f} ({r['test']*100:.2f}%)")
        print(f"  Best Params  : {r['params']}")


if __name__ == "__main__":
    main()
