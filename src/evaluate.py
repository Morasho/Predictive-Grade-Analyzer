"""
evaluate.py — Evaluation metrics, confusion matrix, and feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate_classification(model, X_test, y_test, class_names=None, output_dir="outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    # Use only classes present in either y_test or y_pred
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [class_names[i] for i in present_labels] if class_names else [str(i) for i in present_labels]

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix | Accuracy: {acc:.2%}")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[SAVED] Confusion matrix → {cm_path}")

    return acc, y_pred


def evaluate_regression(model, X_test, y_test, output_dir="outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"REGRESSION RESULTS")
    print(f"{'='*50}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, y_test, alpha=0.4, color="steelblue", edgecolors="none")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[0].set_xlabel("Predicted Score")
    axes[0].set_ylabel("Actual Score")
    axes[0].set_title(f"Predicted vs Actual | R² = {r2:.4f}")

    axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    reg_path = os.path.join(output_dir, "regression_results.png")
    plt.savefig(reg_path, dpi=150)
    plt.close()
    print(f"[SAVED] Regression plots → {reg_path}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_feature_importance(model, feature_names, output_dir="outputs/", model_name="xgboost"):
    os.makedirs(output_dir, exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        print(f"[SKIP] {model_name} does not support feature importance plots.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color="steelblue")
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"{model_name.upper()} — Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"[SAVED] Feature importance → {fi_path}")
