"""
debug_model.py — Run this from your project root to diagnose the model
Usage: python debug_model.py
"""
import joblib
import numpy as np
from pathlib import Path

print("=" * 55)
print("MODEL DIAGNOSIS")
print("=" * 55)

# Try loading the model
model_path = Path("outputs/xgboost_classifier.pkl")
tuned_path  = Path("outputs/xgboost_tuned_classifier.pkl")

path = tuned_path if tuned_path.exists() else model_path
if not path.exists():
    print("ERROR: No model found in outputs/")
    print("Run: python main.py --task both --model both")
    exit(1)

print(f"Loading: {path}")
clf = joblib.load(path)

print(f"\nModel type   : {type(clf).__name__}")
print(f"Classes_     : {clf.classes_}")
print(f"Class dtype  : {type(clf.classes_[0])}")

# Test 1: Strong student — should be Pass
X_strong = np.array([[22, 0.97, 3.9, 99, 6, 8.0, 0, 2, 9,
                       22*0.97, 22*0.4+99*0.3+6*0.3, 8.0-0*2]])
pred_strong = clf.predict(X_strong)[0]
prob_strong = clf.predict_proba(X_strong)[0]
print(f"\nStrong student prediction : {pred_strong}")
print(f"Probabilities             : {dict(zip(clf.classes_, prob_strong.round(3)))}")

# Test 2: Weak student — should be At-Risk
X_weak = np.array([[4, 0.45, 1.2, 50, 0, 4.5, 1, 0, 2,
                    4*0.45, 4*0.4+50*0.3+0*0.3, 4.5-1*2]])
pred_weak = clf.predict(X_weak)[0]
prob_weak = clf.predict_proba(X_weak)[0]
print(f"\nWeak student prediction   : {pred_weak}")
print(f"Probabilities             : {dict(zip(clf.classes_, prob_weak.round(3)))}")

print("\n" + "=" * 55)
print("VERDICT")
print("=" * 55)
if str(pred_strong) == str(pred_weak):
    print("PROBLEM: Both students get same prediction — model is biased.")
    print("ACTION : You need to retrain. Run: python fix_retrain.py")
else:
    print("Model predicts different classes correctly.")
    print("The issue is just the label mapping in api.py.")
    print(f"Your actual class labels are: {list(clf.classes_)}")
    print("Update LABEL_MAP in api.py to match these exactly.")