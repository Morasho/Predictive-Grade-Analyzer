"""
api.py — FastAPI backend for Predictive Grade Analyzer

Run locally:
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000           → single student predictor
    http://localhost:8000/dashboard → educator class dashboard
"""

import os
import sys
import io
import csv
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))
from src.preprocess import FEATURE_COLS
from src.explain import explain_prediction, top_factors, FEATURE_NAMES

MODEL_DIR = "outputs/"
models = {}


# ── Feature engineering (shared) ─────────────────────────────────────────────

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


def load_scaler(name: str):
    """Load scaler for a model if one exists."""
    path = Path(MODEL_DIR) / f"{name}_scaler.pkl"
    if path.exists():
        return joblib.load(path)
    return None


# ── Shared prediction logic (used by /predict AND /predict-batch) ─────────────

def run_prediction(payload: dict, model_name: str) -> dict:
    """
    Core prediction logic reused by both single and batch endpoints.
    payload: dict with the 9 raw feature keys (no model_name key).
    Returns the same structure as the /predict response.
    """
    if model_name not in models:
        available = list(models.keys())
        if not available:
            raise HTTPException(503, "No models loaded. Run main.py first.")
        model_name = available[0]

    clf_model = models[model_name]["classifier"]
    reg_model = models[model_name]["regressor"]

    features = dict(payload)  # copy so we don't mutate the original
    X_raw = engineer_features(features)

    # Apply scaler if one was saved with this model
    scaler = models[model_name].get("scaler")
    X = scaler.transform(X_raw) if scaler is not None else X_raw

    # Normalise labels — model may store strings ("At-Risk") or ints (0/1/2)
    LABEL_MAP = {"0": "At-Risk", "1": "Average", "2": "Pass"}

    raw_grade   = str(clf_model.predict(X)[0])
    grade       = LABEL_MAP.get(raw_grade, raw_grade)
    proba       = clf_model.predict_proba(X)[0]
    score       = float(reg_model.predict(X)[0])
    raw_classes = [str(c) for c in clf_model.classes_]
    classes     = [LABEL_MAP.get(c, c) for c in raw_classes]
    confidence  = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    try:
        # Always pass scaled X so fallback perturbation works for MLP too
        contributions = explain_prediction(clf_model, X, classes)
        factors = top_factors(contributions, n=5)
        if not isinstance(factors, dict):
            factors = {"helping": [], "hurting": []}
        factors.setdefault("helping", [])
        factors.setdefault("hurting", [])
    except Exception as e:
        print(f"[WARN] Explanation failed: {e}")
        contributions = []
        factors = {"helping": [], "hurting": []}

    return {
        "grade":       grade,
        "score":       round(score, 1),
        "confidence":  confidence,
        "model_used":  model_name,
        "explanation": {
            "top_factors":       factors,
            "all_contributions": contributions[:8],
        },
    }


# ── Intervention engine ───────────────────────────────────────────────────────

INTERVENTION_RULES = {
    "study_hours_per_week": {
        "threshold": 15,
        "direction": "low",
        "advice": "Encourage at least 15 study hours/week. Consider a structured weekly study plan."
    },
    "attendance_rate": {
        "threshold": 0.80,
        "direction": "low",
        "advice": "Attendance is below 80%. Regular attendance strongly predicts performance — flag for follow-up."
    },
    "previous_gpa": {
        "threshold": 2.5,
        "direction": "low",
        "advice": "Prior GPA is below 2.5. Early academic support or tutoring referral is recommended."
    },
    "assignments_completed": {
        "threshold": 75,
        "direction": "low",
        "advice": "Less than 75% of assignments submitted. Check for workload issues or disengagement."
    },
    "tutoring_sessions": {
        "threshold": 2,
        "direction": "low",
        "advice": "Fewer than 2 tutoring sessions this term. Encourage booking sessions with the academic support centre."
    },
    "sleep_hours": {
        "threshold": 6.5,
        "direction": "low",
        "advice": "Averaging under 6.5 hours of sleep. Poor sleep significantly impacts cognition — refer to wellness resources."
    },
    "part_time_job": {
        "threshold": 1,
        "direction": "high",
        "advice": "Student holds a part-time job. Monitor for signs of fatigue or time pressure affecting studies."
    },
}


def generate_interventions(student_data: dict, grade: str, shap_factors: dict) -> list:
    """
    Generate actionable intervention suggestions for a student based on
    their raw feature values and SHAP explanation.
    """
    interventions = []

    # Priority 1: SHAP-driven — what's actually hurting this student most
    hurting = shap_factors.get("hurting", [])
    for factor in hurting:
        feature_key = factor.get("label", "").lower().replace(" ", "_").replace("-", "_")
        for rule_key, rule in INTERVENTION_RULES.items():
            if rule_key in feature_key or feature_key in rule_key:
                interventions.append({
                    "priority":    "high",
                    "feature":     factor["label"],
                    "advice":      rule["advice"],
                    "shap_impact": round(abs(factor.get("shap_value", 0)), 3)
                })
                break

    # Priority 2: Threshold-based — catch anything SHAP didn't surface
    for feature, rule in INTERVENTION_RULES.items():
        value = student_data.get(feature)
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        triggered = (
            (rule["direction"] == "low"  and value < rule["threshold"]) or
            (rule["direction"] == "high" and value >= rule["threshold"])
        )

        if triggered:
            already_added = any(
                i["feature"].lower().replace(" ", "_") in feature
                for i in interventions
            )
            if not already_added:
                interventions.append({
                    "priority":    "medium",
                    "feature":     feature.replace("_", " ").title(),
                    "advice":      rule["advice"],
                    "shap_impact": None
                })

    # If passing with no issues, give a positive note
    if grade == "Pass" and not interventions:
        interventions.append({
            "priority":    "info",
            "feature":     "Overall",
            "advice":      "Student is on track. Encourage continued engagement and highlight strengths.",
            "shap_impact": None
        })

    return interventions[:5]  # cap at 5 per student


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("xgboost", "mlp"):
        clf_m  = load_best_model(name, "classifier")
        reg_m  = load_best_model(name, "regressor")
        scaler = load_scaler(name)
        if clf_m and reg_m:
            models[name] = {"classifier": clf_m, "regressor": reg_m, "scaler": scaler}
            print(f"[INFO] Loaded {name} models" + (" + scaler" if scaler else ""))
    if not models:
        print("[WARN] No models found — run main.py first")
    yield


app = FastAPI(
    title="Predictive Grade Analyzer API",
    description="""
## 🎓 Predictive Grade Analyzer

An ML pipeline that predicts student academic performance using **XGBoost** and **MLP Neural Networks**.

### Features
- **Single prediction** — predict one student's grade with SHAP explanation
- **Batch prediction** — upload a class CSV and get risk analysis for every student
- **Intervention suggestions** — actionable advice for at-risk students
- **Two models** — XGBoost (78.75% accuracy) and MLP Neural Network (81.25% accuracy)

### Grade Categories
| Grade | Description |
|-------|-------------|
| `Pass` | Top 35% performers |
| `Average` | Middle 40% performers |
| `At-Risk` | Bottom 25% — needs intervention |
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Victor Morara",
        "url": "https://github.com/Morasho",
        "email": "moraravictor9@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
)


# ── Schemas ───────────────────────────────────────────────────────────────────

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
    model_name:            str   = Field("xgboost",   example="xgboost")


# ── Required CSV columns for batch upload ─────────────────────────────────────

REQUIRED_FEATURES = [
    "study_hours_per_week",
    "attendance_rate",
    "previous_gpa",
    "assignments_completed",
    "tutoring_sessions",
    "sleep_hours",
    "part_time_job",
    "extracurriculars",
    "socioeconomic_index",
]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    dashboard_path = base_dir / "static" / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"dashboard.html not found at {dashboard_path}. "
                   "Copy dashboard.html into your static/ folder first."
        )
    return FileResponse(str(dashboard_path))


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Check API status and which models are currently loaded."""
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.get("/models", summary="List available models", tags=["System"])
async def list_models():
    """Returns the list of ML models currently loaded and ready for predictions."""
    return {"available": list(models.keys())}


@app.post(
    "/predict",
    summary="Predict single student performance",
    tags=["Prediction"],
    response_description="Grade prediction with confidence scores and SHAP explanation",
)
async def predict(data: StudentInput):
    """
    Predict a single student's academic performance.

    Returns:
    - **grade**: `Pass`, `Average`, or `At-Risk`
    - **score**: Predicted score out of 100
    - **confidence**: Probability breakdown across all three classes
    - **explanation**: Top SHAP factors helping and hurting the prediction
    - **model_used**: Which model produced the result
    """
    payload = data.model_dump()
    payload.pop("model_name")
    return run_prediction(payload, data.model_name.lower())


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.post(
    "/predict-batch",
    summary="Predict performance for a whole class",
    tags=["Prediction"],
    response_description="Class summary with per-student predictions, SHAP factors and interventions",
)
async def predict_batch(file: UploadFile = File(...), model_name: str = "xgboost"):
    """
    Upload a CSV file with one student per row to get predictions for the entire class.

    **Required CSV columns:**
    `student_name` *(optional)*, `study_hours_per_week`, `attendance_rate`, `previous_gpa`,
    `assignments_completed`, `tutoring_sessions`, `sleep_hours`, `part_time_job`,
    `extracurriculars`, `socioeconomic_index`

    **Download a template:** `GET /predict-batch/template`

    Returns:
    - **summary**: Class-wide grade distribution and pass rate
    - **students**: Per-student predictions, confidence scores, SHAP factors and intervention suggestions
    """
    # 1. Parse the CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {missing}"
        )

    if "student_name" not in df.columns:
        df["student_name"] = [f"Student {i + 1}" for i in range(len(df))]

    # 2. Predict per student
    results = []
    for idx, row in df.iterrows():
        student_name = str(row.get("student_name", f"Student {idx + 1}"))
        payload = {feat: row[feat] for feat in REQUIRED_FEATURES}

        try:
            prediction  = run_prediction(payload, model_name)
            grade       = prediction["grade"]
            score       = prediction.get("score", 0)
            confidence  = prediction["confidence"]
            shap_factors = prediction.get("explanation", {}).get("top_factors", {})

            interventions = generate_interventions(
                student_data=payload,
                grade=grade,
                shap_factors=shap_factors,
            )

            results.append({
                "student_name": student_name,
                "grade":        grade,
                "score":        round(score, 1),
                "confidence":   confidence,
                "interventions": interventions,
                "shap_factors": shap_factors,
                "raw_features": {k: float(v) for k, v in payload.items()},
                "row_index":    int(idx),
            })

        except Exception as e:
            results.append({
                "student_name": student_name,
                "grade":        "Error",
                "score":        0,
                "confidence":   {},
                "interventions": [],
                "shap_factors": {},
                "raw_features": {},
                "row_index":    int(idx),
                "error":        str(e),
            })

    # 3. Class-level summary
    valid = [r for r in results if r["grade"] != "Error"]
    grade_counts = {"Pass": 0, "Average": 0, "At-Risk": 0}
    for r in valid:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

    summary = {
        "total_students":    len(results),
        "valid_predictions": len(valid),
        "grade_distribution": grade_counts,
        "at_risk_count":     grade_counts.get("At-Risk", 0),
        "pass_rate":         round(grade_counts.get("Pass", 0) / max(len(valid), 1) * 100, 1),
        "model_used":        model_name,
    }

    return {"summary": summary, "students": results}


@app.get(
    "/predict-batch/template",
    summary="Download CSV template",
    tags=["Prediction"],
)
async def batch_template():
    """
    Download a blank CSV template with the correct column headers and two sample rows.
    Fill this in with your class data and upload to `/predict-batch`.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["student_name"] + REQUIRED_FEATURES)
    writer.writerow(["Alice Njeri",   18, 0.92, 3.4, 90, 4, 7.5, 0, 2, 8])
    writer.writerow(["Brian Omondi",   8, 0.60, 1.9, 55, 1, 5.5, 1, 1, 4])
    writer.writerow(["Carol Wangari", 12, 0.75, 2.8, 70, 2, 7.0, 0, 3, 6])
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=student_template.csv"},
    )


# ── Static files (must be mounted LAST so route handlers take priority) ───────
# If mounted before routes, FastAPI's StaticFiles intercepts /dashboard
# before the route handler can respond — mounting last fixes that.
app.mount("/static", StaticFiles(directory="static"), name="static")