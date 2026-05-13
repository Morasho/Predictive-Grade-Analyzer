"""
debug_explain.py — Check what explain_prediction returns for MLP
Run from project root: python debug_explain.py
"""
import sys, os, joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))
from src.explain import explain_prediction, top_factors, FEATURE_NAMES

# Load MLP
mlp_clf = joblib.load("outputs/mlp_classifier.pkl")
scaler  = joblib.load("outputs/mlp_scaler.pkl")

# Default student (same as form)
raw = np.array([[15, 0.85, 3.0, 85, 3, 7.0, 0, 2, 6,
                 15*0.85, 15*0.4+85*0.3+3*0.3, 7.0-0*2]])
X_scaled = scaler.transform(raw)

print("Model type:", type(mlp_clf).__name__)
print("predict_proba:", mlp_clf.predict_proba(X_scaled)[0])
print("predict:", mlp_clf.predict(X_scaled)[0])
print("classes_:", mlp_clf.classes_)
print()

# Test the fallback directly
base_proba = mlp_clf.predict_proba(X_scaled)[0]
predicted_class_idx = int(np.argmax(base_proba))
print(f"Predicted class index: {predicted_class_idx}")
print(f"Base proba for that class: {base_proba[predicted_class_idx]:.6f}")
print()

# Check sensitivity of each feature
print("Feature sensitivities (epsilon=1e-2):")
for i, name in enumerate(FEATURE_NAMES):
    X_up = X_scaled.copy()
    X_up[0, i] += 1e-2
    proba_up = mlp_clf.predict_proba(X_up)[0]
    delta = float(proba_up[predicted_class_idx] - base_proba[predicted_class_idx])
    sensitivity = round(delta / 1e-2, 4)
    print(f"  {name:35s}: delta={delta:.6f}, sensitivity={sensitivity}")

print()
contributions = explain_prediction(mlp_clf, X_scaled, list(mlp_clf.classes_))
print(f"contributions returned: {len(contributions)} items")
for c in contributions[:5]:
    print(f"  {c}")

factors = top_factors(contributions)
print(f"\ntop_factors: {factors}")