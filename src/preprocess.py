"""
preprocess.py — Data loading, cleaning, feature engineering, and class balancing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
import os


FEATURE_COLS = [
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

REGRESSION_TARGET = "final_score"
CLASSIFICATION_TARGET = "grade"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} records | Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["student_id"], errors="ignore", inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in FEATURE_COLS:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q3)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_attendance_ratio"] = df["study_hours_per_week"] * df["attendance_rate"]
    df["academic_effort_score"] = (
        df["study_hours_per_week"] * 0.4
        + df["assignments_completed"] * 0.3
        + df["tutoring_sessions"] * 0.3
    )
    df["lifestyle_balance"] = df["sleep_hours"] - df["part_time_job"] * 2
    return df


def random_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """
    Oversample minority classes by duplicating samples with small Gaussian noise.
    Only applied to training data, never to test data.
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled, y_resampled = [X], [y]

    for cls, count in zip(classes, counts):
        if count < max_count:
            shortfall = max_count - count
            idx = np.where(y == cls)[0]
            chosen = rng.choice(idx, size=shortfall, replace=True)
            noise = rng.normal(0, 0.05, X[chosen].shape)
            X_resampled.append(X[chosen] + noise)
            y_resampled.append(np.full(shortfall, cls))

    X_out = np.vstack(X_resampled)
    y_out = np.concatenate(y_resampled)

    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def prepare_data(path: str, task: str = "classification", test_size: float = 0.2,
                 oversample: bool = True):
    """
    Full preprocessing pipeline.

    Args:
        path: Path to CSV file
        task: 'regression' or 'classification'
        test_size: Fraction for test split
        oversample: Apply random oversampling on training set (classification only)

    Returns:
        X_train, X_test, y_train, y_test, scaler, le, sample_weights
    """
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    feature_cols = FEATURE_COLS + [
        "study_attendance_ratio",
        "academic_effort_score",
        "lifestyle_balance",
    ]

    X = df[[c for c in feature_cols if c in df.columns]]

    le = None
    sample_weights = None

    if task == "regression":
        y = df[REGRESSION_TARGET]
    elif task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(df[CLASSIFICATION_TARGET])
        print(f"[INFO] Classes: {list(le.classes_)}")
        classes, counts = np.unique(y, return_counts=True)
        print(f"[INFO] Class distribution: { {le.classes_[i]: int(c) for i, c in zip(classes, counts)} }")
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task == "classification" else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if task == "classification":
        if oversample:
            X_train_scaled, y_train = random_oversample(X_train_scaled, y_train)
            print(f"[INFO] After oversampling — Train: {X_train_scaled.shape}")
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    print(f"[INFO] Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, sample_weights
