"""
models.py — XGBoost and MLP model definitions
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator


def get_model(model_name: str, task: str = "classification") -> BaseEstimator:
    """
    Factory function to retrieve a model by name and task.

    Args:
        model_name: 'xgboost' or 'mlp'
        task: 'classification' or 'regression'

    Returns:
        Configured sklearn-compatible estimator
    """
    models = {
        "classification": {
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,      # prevents overfitting on tiny classes
                gamma=0.1,               # min loss reduction for split
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=0.005,             # stronger regularisation
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            ),
        },
        "regression": {
            "xgboost": XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
        },
    }

    if task not in models:
        raise ValueError(f"Unknown task: '{task}'. Choose 'classification' or 'regression'.")
    if model_name not in models[task]:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(models[task].keys())}")

    return models[task][model_name]
