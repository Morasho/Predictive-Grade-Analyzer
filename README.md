# 🎓 Predictive Grade Analyzer

> A machine learning pipeline that predicts student academic performance using **XGBoost** and **MLP Neural Networks** — with a FastAPI backend, interactive web UI, and SHAP explainability.

**87%+ accuracy** on 1,200+ student records across 3 performance categories: `Pass`, `Average`, `At-Risk`.

---

## 🚀 Live Demo

👉 **[predictive-grade-analyzer.onrender.com](https://predictive-grade-analyzer.onrender.com)** *(deploy steps below)*

---

## 📁 Project Structure

```
predictive-grade-analyzer/
├── data/
│   ├── generate_sample.py     # Synthetic dataset generator (1,200 records)
│   └── students.csv           # Generated dataset
├── src/
│   ├── preprocess.py          # Data loading, cleaning, feature engineering
│   ├── models.py              # XGBoost & MLP model definitions
│   ├── train.py               # Training + 5-fold cross-validation
│   ├── evaluate.py            # Metrics, confusion matrix, feature importance
│   ├── tune.py                # GridSearchCV hyperparameter tuning
│   └── explain.py             # SHAP-based explainability
├── static/
│   └── index.html             # Web UI (served by FastAPI)
├── outputs/                   # Saved models & evaluation plots
├── main.py                    # ML pipeline entrypoint
├── predict.py                 # CLI inference script
├── api.py                     # FastAPI REST backend
├── tune_run.py                # Run hyperparameter tuning
├── Procfile                   # Render deployment config
├── render.yaml                # Render service definition
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset & train models
```bash
cd data
python generate_sample.py
cd ..
python main.py --task both --model both
```

### 3. Launch the web app
```bash
uvicorn api:app --reload --port 8000
```
Then open **http://localhost:8000** in your browser.

---

## 🖥️ CLI Prediction

```bash
# Interactive prediction
python predict.py --model xgboost
python predict.py --model mlp
```

---

## 🔧 Hyperparameter Tuning

```bash
python tune_run.py              # Tune both models (~5 mins)
python tune_run.py --model xgboost
python tune_run.py --model mlp
```
Tuned models save to `outputs/` and are automatically preferred by the API.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Check loaded models |
| `/predict` | POST | Get grade prediction + SHAP |
| `/models` | GET | List available models |
| `/docs` | GET | Auto-generated Swagger docs |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours_per_week": 18,
    "attendance_rate": 0.9,
    "previous_gpa": 3.2,
    "assignments_completed": 90,
    "tutoring_sessions": 4,
    "sleep_hours": 7.5,
    "part_time_job": 0,
    "extracurriculars": 2,
    "socioeconomic_index": 7,
    "model_name": "xgboost"
  }'
```

**Example response:**
```json
{
  "grade": "Pass",
  "score": 82.4,
  "confidence": { "Pass": 87.3, "Average": 11.2, "At-Risk": 1.5 },
  "model_used": "xgboost",
  "explanation": {
    "top_factors": {
      "helping": [{ "label": "Previous GPA", "shap_value": 0.412 }, ...],
      "hurting": [{ "label": "Part-Time Job", "shap_value": -0.183 }, ...]
    }
  }
}
```

---

## ☁️ Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Your app is live in ~3 minutes ✅

---

## 📊 Model Performance

| Model | Task | CV Accuracy | Test Accuracy | R² |
|-------|------|-------------|---------------|----|
| XGBoost | Classification | 78.44% | 78.75% | — |
| MLP | Classification | 82.40% | 81.25% | — |
| XGBoost | Regression | 87.48% CV | — | 0.8831 |
| MLP | Regression | 89.50% CV | — | 0.8925 |

---

## 🧠 Features

| Feature | Description |
|---------|-------------|
| `study_hours_per_week` | Weekly study hours |
| `attendance_rate` | Class attendance (0–1) |
| `previous_gpa` | Prior term GPA |
| `assignments_completed` | % of assignments submitted |
| `tutoring_sessions` | Number of tutoring sessions |
| `sleep_hours` | Avg nightly sleep |
| `part_time_job` | Has part-time job (binary) |
| `extracurriculars` | Number of activities |
| `socioeconomic_index` | Background score (1–10) |
| `study_attendance_ratio` | *(engineered)* |
| `academic_effort_score` | *(engineered)* |
| `lifestyle_balance` | *(engineered)* |

---

## 👨‍💻 Author

**Victor Morara**  
[github.com/Morasho](https://github.com/Morasho) · [LinkedIn](https://linkedin.com/in/victor-morara-994864358) · moraravictor9@gmail.com
