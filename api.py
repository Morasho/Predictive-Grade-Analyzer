"""
api.py — FastAPI backend for Predictive Grade Analyzer

Run locally:
    uvicorn api:app --reload --port 8000

Then open: http://localhost:8000
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS
from src.explain import explain_prediction, top_factors, FEATURE_NAMES

MODEL_DIR = "outputs/"
models = {}


def engineer_features(data: dict) -> np.ndarray:
    study  = data["study_hours_per_week"]
    attend = data["attendance_rate"]
    assign = data["assignments_completed"]
    tutor  = data["tutoring_sessions"]
    sleep  = data["sleep_hours"]
    job    = data["part_time_job"]

    data["study_attendance_ratio"] = study * attend
    data["academic_effort_score"]  = study * 0.4 + assign * 0.3 + tutor * 0.3
    data["lifestyle_balance"]      = sleep - job * 2

    cols = FEATURE_COLS + ["study_attendance_ratio", "academic_effort_score", "lifestyle_balance"]
    return np.array([[data[c] for c in cols]])


def load_best_model(name: str, kind: str):
    """Load tuned model if available, else fall back to base model."""
    tuned = Path(MODEL_DIR) / f"{name}_tuned_{kind}.pkl"
    base  = Path(MODEL_DIR) / f"{name}_{kind}.pkl"
    if tuned.exists():
        return joblib.load(tuned)
    if base.exists():
        return joblib.load(base)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("xgboost", "mlp"):
        clf = load_best_model(name, "classifier")
        reg = load_best_model(name, "regressor")
        if clf and reg:
            models[name] = {"classifier": clf, "regressor": reg}
            print(f"[INFO] Loaded {name} models")
    if not models:
        print("[WARN] No models found — run main.py first")
    yield


app = FastAPI(
    title="Predictive Grade Analyzer API",
    description="ML pipeline predicting student performance using XGBoost and MLP",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Schemas ──────────────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    study_hours_per_week:  float = Field(..., ge=1,   le=40,  example=15.0)
    attendance_rate:       float = Field(..., ge=0.0, le=1.0, example=0.85)
    previous_gpa:          float = Field(..., ge=0.0, le=4.0, example=3.0)
    assignments_completed: float = Field(..., ge=50,  le=100, example=85.0)
    tutoring_sessions:     float = Field(..., ge=0,   le=20,  example=3.0)
    sleep_hours:           float = Field(..., ge=4,   le=10,  example=7.0)
    part_time_job:         int   = Field(..., ge=0,   le=1,   example=0)
    extracurriculars:      float = Field(..., ge=0,   le=8,   example=2.0)
    socioeconomic_index:   float = Field(..., ge=1,   le=10,  example=6.0)
    model_name:            str   = Field("xgboost", example="xgboost")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.post("/predict")
async def predict(data: StudentInput):
    model_name = data.model_name.lower()
    if model_name not in models:
        available = list(models.keys())
        if not available:
            raise HTTPException(503, "No models loaded. Run main.py first.")
        model_name = available[0]

    clf = models[model_name]["classifier"]
    reg = models[model_name]["regressor"]

    features = data.model_dump()
    features.pop("model_name")
    X = engineer_features(features)

    grade      = clf.predict(X)[0]
    proba      = clf.predict_proba(X)[0]
    score      = float(reg.predict(X)[0])
    classes    = list(clf.classes_)
    confidence = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    # SHAP explanation
    contributions = explain_prediction(clf, X, classes)
    factors = top_factors(contributions, n=5)

    return {
        "grade":        grade,
        "score":        round(score, 1),
        "confidence":   confidence,
        "model_used":   model_name,
        "explanation":  {
            "top_factors": factors,
            "all_contributions": contributions[:8],
        },
    }


@app.get("/models")
async def list_models():
    return {"available": list(models.keys())}
