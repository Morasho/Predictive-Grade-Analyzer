"""
explain.py — SHAP-based explainability for XGBoost models
"""

import numpy as np
import os

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

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For multiclass, shap_values is a list — pick predicted class
        predicted_class_idx = int(model.predict(X)[0])

        if isinstance(shap_values, list):
            sv = shap_values[predicted_class_idx][0]
        else:
            sv = shap_values[0]

        contributions = []
        for i, name in enumerate(FEATURE_NAMES):
            val = float(sv[i])
            contributions.append({
                "feature":    name,
                "label":      FEATURE_LABELS.get(name, name),
                "shap_value": round(val, 4),
                "direction":  "positive" if val > 0 else "negative",
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    except ImportError:
        # Graceful fallback using feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            contributions = []
            for i, name in enumerate(FEATURE_NAMES):
                val = float(importances[i])
                contributions.append({
                    "feature":    name,
                    "label":      FEATURE_LABELS.get(name, name),
                    "shap_value": round(val, 4),
                    "direction":  "positive",
                })
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return contributions
        return []


def top_factors(contributions: list, n: int = 5) -> dict:
    """Return top n positive and negative factors."""
    positive = [c for c in contributions if c["direction"] == "positive"][:n]
    negative = [c for c in contributions if c["direction"] == "negative"][:n]
    return {"helping": positive, "hurting": negative}
