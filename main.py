"""
main.py — Predictive Grade Analyzer
Entry point for the full ML pipeline.

Usage:
    python main.py --task classification --model xgboost
    python main.py --task regression --model mlp
    python main.py --task both --model both
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.preprocess import prepare_data
from src.train import train_model, save_model
from src.evaluate import (
    evaluate_classification,
    evaluate_regression,
    plot_feature_importance,
)

DATA_PATH = "data/students.csv"
OUTPUT_DIR = "outputs/"

FEATURE_NAMES = [
    "study_hours_per_week",
    "attendance_rate",
    "previous_gpa",
    "assignments_completed",
    "tutoring_sessions",
    "sleep_hours",
    "part_time_job",
    "extracurriculars",
    "socioeconomic_index",
    "study_attendance_ratio",
    "academic_effort_score",
    "lifestyle_balance",
]


def run_classification(model_name: str):
    print("\n🎓 Running CLASSIFICATION pipeline...")
    # Data is already balanced — no oversampling needed, class_weight handles it
    X_train, X_test, y_train, y_test, scaler, le, sample_weights = prepare_data(
        DATA_PATH, task="classification", oversample=False
    )
    class_names = list(le.classes_)
    model, cv_scores = train_model(
        X_train, y_train, model_name, task="classification",
        sample_weights=sample_weights
    )
    acc, _ = evaluate_classification(
        model, X_test, y_test, class_names=class_names, output_dir=OUTPUT_DIR
    )
    if model_name == "xgboost":
        plot_feature_importance(model, FEATURE_NAMES, output_dir=OUTPUT_DIR, model_name=model_name)
    save_model(model, f"{model_name}_classifier", output_dir=OUTPUT_DIR)
    return acc


def run_regression(model_name: str):
    print("\n📈 Running REGRESSION pipeline...")
    X_train, X_test, y_train, y_test, scaler, le, _ = prepare_data(
        DATA_PATH, task="regression"
    )
    model, cv_scores = train_model(X_train, y_train, model_name, task="regression")
    metrics = evaluate_regression(model, X_test, y_test, output_dir=OUTPUT_DIR)
    save_model(model, f"{model_name}_regressor", output_dir=OUTPUT_DIR)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Predictive Grade Analyzer")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    parser.add_argument("--model", choices=["xgboost", "mlp", "both"], default="both")
    args = parser.parse_args()

    models = ["xgboost", "mlp"] if args.model == "both" else [args.model]
    tasks = ["classification", "regression"] if args.task == "both" else [args.task]

    results = {}
    for model_name in models:
        for task in tasks:
            key = f"{model_name}_{task}"
            if task == "classification":
                results[key] = run_classification(model_name)
            else:
                results[key] = run_regression(model_name)

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"{key}: R²={val['r2']:.4f} | RMSE={val['rmse']:.4f}")
        else:
            print(f"{key}: Accuracy={val:.4f} ({val*100:.2f}%)")


if __name__ == "__main__":
    main()
