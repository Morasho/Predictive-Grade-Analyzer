import pandas as pd
import numpy as np

np.random.seed(42)
n = 1200

data = {
    "student_id": range(1, n + 1),
    "study_hours_per_week": np.random.normal(15, 5, n).clip(1, 40),
    "attendance_rate": np.random.normal(0.80, 0.12, n).clip(0.3, 1.0),
    "previous_gpa": np.random.normal(2.8, 0.6, n).clip(0.0, 4.0),
    "assignments_completed": np.random.randint(60, 101, n),
    "tutoring_sessions": np.random.poisson(3, n),
    "sleep_hours": np.random.normal(7, 1.2, n).clip(4, 10),
    "part_time_job": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    "extracurriculars": np.random.randint(0, 5, n),
    "socioeconomic_index": np.random.uniform(1, 10, n),
}

df = pd.DataFrame(data)

raw_score = (
    df["study_hours_per_week"] * 1.2
    + df["attendance_rate"] * 30
    + df["previous_gpa"] * 10
    + df["assignments_completed"] * 0.3
    + df["tutoring_sessions"] * 1.5
    - df["part_time_job"] * 3
    + np.random.normal(0, 3, n)
)

# Normalize to full 0-100 range
df["final_score"] = (
    (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min()) * 100
).round(2)

# 3-class grading: Pass (top 35%), Average (middle 40%), At-Risk (bottom 25%)
def assign_grade(p):
    if p <= 0.25:   return "At-Risk"
    elif p <= 0.65: return "Average"
    else:           return "Pass"

df["grade"] = df["final_score"].rank(pct=True, method="first").apply(assign_grade)

df.to_csv("students.csv", index=False)
print(f"Dataset saved: {len(df)} records")
print(df["grade"].value_counts().sort_index())
print(f"\nScore range: {df['final_score'].min():.1f} - {df['final_score'].max():.1f}")
