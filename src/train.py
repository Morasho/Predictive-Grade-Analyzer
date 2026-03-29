"""
train.py — Model training with cross-validation and optional sample weighting
"""

import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from src.models import get_model


def train_model(X_train, y_train, model_name: str, task: str = "classification",
                sample_weights=None):
    """
    Train a model with 5-fold cross-validation.

    Args:
        X_train: Training features (scaled)
        y_train: Training labels/targets
        model_name: 'xgboost' or 'mlp'
        task: 'classification' or 'regression'
        sample_weights: Optional array of per-sample weights (classification only)

    Returns:
        Fitted model, cross-val scores
    """
    print(f"\n{'='*50}")
    print(f"Training: {model_name.upper()} | Task: {task}")
    print(f"{'='*50}")

    model = get_model(model_name, task)

    if task == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "r2"

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"[CV] {scoring.upper()} Scores: {np.round(cv_scores, 4)}")
    print(f"[CV] Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit — pass sample_weight for XGBoost if available
    if sample_weights is not None and model_name == "xgboost":
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    print(f"[INFO] Model training complete.")
    return model, cv_scores


def save_model(model, name: str, output_dir: str = "outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[SAVED] Model saved to: {path}")
    return path


def load_model(path: str):
    model = joblib.load(path)
    print(f"[LOADED] Model loaded from: {path}")
    return model
