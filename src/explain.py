"""
explain.py — SHAP-based explainability for XGBoost models
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Model type not yet supported")

FEATURE_NAMES = [
    "study_hours_per_week", "attendance_rate", "previous_gpa",
    "assignments_completed", "tutoring_sessions", "sleep_hours",
    "part_time_job", "extracurriculars", "socioeconomic_index",
    "study_attendance_ratio", "academic_effort_score", "lifestyle_balance",
]

FEATURE_LABELS = {
    "study_hours_per_week":   "Study Hours/Week",
    "attendance_rate":        "Attendance Rate",
    "previous_gpa":           "Previous GPA",
    "assignments_completed":  "Assignments Completed",
    "tutoring_sessions":      "Tutoring Sessions",
    "sleep_hours":            "Sleep Hours",
    "part_time_job":          "Part-Time Job",
    "extracurriculars":       "Extracurriculars",
    "socioeconomic_index":    "Socioeconomic Index",
    "study_attendance_ratio": "Study × Attendance",
    "academic_effort_score":  "Academic Effort Score",
    "lifestyle_balance":      "Lifestyle Balance",
}


def explain_prediction(model, X: np.ndarray, class_names: list) -> list:
    """
    Generate SHAP values for a single prediction.
    Falls back to feature importances if SHAP is not installed.
    Returns a list of dicts: [{feature, label, shap_value, direction}, ...]
    sorted by absolute impact descending.
    """
    try:
        import shap

        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Resolve predicted class index — works for both int and string labels
        raw_pred = model.predict(X)[0]
        classes  = list(model.classes_)

        try:
            # Model predicts string labels e.g. "At-Risk"
            predicted_class_idx = classes.index(raw_pred)
        except (ValueError, TypeError):
            # Model predicts numeric labels e.g. 0, 1, 2
            try:
                predicted_class_idx = int(raw_pred)
            except (ValueError, TypeError):
                predicted_class_idx = 0

        # Normalise shap_values into a flat 1-D array for the predicted class
        sv = None
        if isinstance(shap_values, list):
            # Older SHAP: list of arrays, one per class
            predicted_class_idx = min(predicted_class_idx, len(shap_values) - 1)
            candidate = shap_values[predicted_class_idx]
        else:
            # Newer SHAP: single ndarray, shape (1, n_features) or (1, n_features, n_classes)
            candidate = shap_values

        candidate = np.array(candidate)

        if candidate.ndim == 3:
            # shape (1, n_features, n_classes) — pick predicted class
            predicted_class_idx = min(predicted_class_idx, candidate.shape[2] - 1)
            sv = candidate[0, :, predicted_class_idx]
        elif candidate.ndim == 2:
            # shape (1, n_features)
            sv = candidate[0]
        elif candidate.ndim == 1:
            # already flat
            sv = candidate
        else:
            sv = candidate.flatten()

        contributions = []
        for i, name in enumerate(FEATURE_NAMES):
            if i >= len(sv):
                break
            val = float(np.squeeze(sv[i]))   # squeeze handles 0-d arrays
            contributions.append({
                "feature":    name,
                "label":      FEATURE_LABELS.get(name, name),
                "shap_value": round(val, 4),
                "direction":  "positive" if val > 0 else "negative",
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    except ImportError:
        return _fallback_importance(model, X)

    except Exception as e:
        print(f"[WARN] SHAP explanation error: {e}")
        return _fallback_importance(model, X)


def _fallback_importance(model, X: np.ndarray) -> list:
    """
    Fallback explainability for models that don't support TreeSHAP (e.g. MLP).
    Uses coefficient-of-variation style scoring: perturb each feature slightly
    and measure how much the predicted probabilities shift.
    Works for any sklearn classifier with predict_proba.
    """
    contributions = []

    try:
        base_proba = model.predict_proba(X)[0]
        predicted_class_idx = int(np.argmax(base_proba))
        epsilon = 1e-2

        for i, name in enumerate(FEATURE_NAMES):
            if i >= X.shape[1]:
                break

            # Perturb feature i upward
            X_up = X.copy()
            X_up[0, i] += epsilon

            proba_up = model.predict_proba(X_up)[0]
            delta = float(proba_up[predicted_class_idx] - base_proba[predicted_class_idx])

            contributions.append({
                "feature":    name,
                "label":      FEATURE_LABELS.get(name, name),
                "shap_value": round(delta / epsilon, 4),
                "direction":  "positive" if delta >= 0 else "negative",
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    except Exception as e:
        print(f"[WARN] Fallback importance failed: {e}")

    return contributions


def top_factors(contributions: list, n: int = 5) -> dict:
    """Return top n positive and negative SHAP factors."""
    positive = [c for c in contributions if c["direction"] == "positive"][:n]
    negative = [c for c in contributions if c["direction"] == "negative"][:n]
    return {
        "helping": [{"label": c["label"], "shap_value": c["shap_value"]} for c in positive],
        "hurting": [{"label": c["label"], "shap_value": c["shap_value"]} for c in negative],
    }