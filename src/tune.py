"""
tune.py — Hyperparameter tuning for XGBoost and MLP using GridSearchCV
"""

import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def tune_xgboost(X_train, y_train, output_dir="outputs/"):
    """
    Grid search over XGBoost's most impactful hyperparameters.
    Runs ~108 fits — takes 1–3 minutes.
    """
    print("\n" + "="*50)
    print("TUNING: XGBoost Classifier")
    print("="*50)

    param_grid = {
        "n_estimators":    [200, 300, 400],
        "max_depth":       [4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "min_child_weight":[1, 2, 3],
    }

    base = XGBClassifier(
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"\n[BEST] Params : {search.best_params_}")
    print(f"[BEST] CV Accuracy: {search.best_score_:.4f} ({search.best_score_*100:.2f}%)")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "xgboost_tuned_classifier.pkl")
    joblib.dump(search.best_estimator_, path)
    print(f"[SAVED] Tuned model → {path}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_mlp(X_train, y_train, output_dir="outputs/"):
    """
    Grid search over MLP's most impactful hyperparameters.
    Runs ~60 fits — takes 2–5 minutes.
    """
    print("\n" + "="*50)
    print("TUNING: MLP Classifier")
    print("="*50)

    param_grid = {
        "hidden_layer_sizes": [(128, 64), (256, 128, 64), (128, 64, 32)],
        "alpha":              [0.0001, 0.001, 0.005, 0.01],
        "learning_rate_init": [0.001, 0.005],
        "max_iter":           [500],
    }

    base = MLPClassifier(
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"\n[BEST] Params : {search.best_params_}")
    print(f"[BEST] CV Accuracy: {search.best_score_:.4f} ({search.best_score_*100:.2f}%)")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "mlp_tuned_classifier.pkl")
    joblib.dump(search.best_estimator_, path)
    print(f"[SAVED] Tuned model → {path}")

    return search.best_estimator_, search.best_params_, search.best_score_
