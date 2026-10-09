"""
Data loading and preprocessing utilities for Student Dropout Risk project.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "student_dropout_risk_2026.csv"

CATEGORICAL_COLS = [
    "gender", "nationality", "family_income_level",
    "course_type", "study_mode"
]

BINARY_COLS = [
    "first_generation", "disability_status", "scholarship_holder",
    "tuition_up_to_date", "part_time_job", "international_student",
    "extracurricular_activities", "mental_health_support"
]

NUMERIC_COLS = [
    "age", "weekly_work_hours", "semester", "admission_grade",
    "previous_qualification_grade", "units_enrolled", "units_passed",
    "units_failed", "gpa_semester_1", "gpa_semester_2", "attendance_rate",
    "assignments_submitted_rate", "library_visits_per_month",
    "lms_logins_per_week", "missed_deadlines", "stress_level",
    "satisfaction_score", "commute_time_mins", "sleep_hours"
]

TARGET_COL = "dropout"


@pd.io.common.stringify_path
def _path(p):
    return str(p)


def load_raw_data() -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features.
    Returns a copy with no nulls and engineered columns added.
    """
    df = df.copy()

    # Drop person_id — not a feature
    df.drop(columns=["person_id"], errors="ignore", inplace=True)

    # --- Derived / engineered features ---
    df["pass_rate"] = np.where(
        df["units_enrolled"] > 0,
        df["units_passed"] / df["units_enrolled"],
        0.0
    )
    df["gpa_avg"] = (df["gpa_semester_1"] + df["gpa_semester_2"]) / 2
    df["gpa_change"] = df["gpa_semester_2"] - df["gpa_semester_1"]
    df["engagement_score"] = (
        df["lms_logins_per_week"] * 0.3
        + df["library_visits_per_month"] * 0.2
        + df["attendance_rate"] * 0.3
        + df["assignments_submitted_rate"] * 0.2
    )

    # Encode categoricals with pd.Categorical codes for modelling
    for col in CATEGORICAL_COLS:
        df[col] = pd.Categorical(df[col]).codes

    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (target) as numpy arrays."""
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    return X, y, feature_cols


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of high-level summary statistics."""
    total = len(df)
    dropout_count = int(df[TARGET_COL].sum())
    return {
        "total_students": total,
        "dropout_count": dropout_count,
        "retained_count": total - dropout_count,
        "dropout_rate": round(dropout_count / total * 100, 2),
        "avg_gpa": round(df["gpa_avg"].mean(), 2) if "gpa_avg" in df.columns else round(
            (df["gpa_semester_1"] + df["gpa_semester_2"]).mean() / 2, 2
        ),
        "avg_attendance": round(df["attendance_rate"].mean(), 2),
        "avg_stress": round(df["stress_level"].mean(), 2),
        "avg_satisfaction": round(df["satisfaction_score"].mean(), 2),
    }
