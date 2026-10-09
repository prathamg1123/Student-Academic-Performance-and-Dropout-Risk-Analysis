"""
Model training, evaluation and persistence for Dropout Risk Prediction.
Uses RandomForestClassifier + GradientBoostingClassifier ensemble
(avoids XGBoost native lib crashes on some Python 3.14 builds).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH    = MODELS_DIR / "rf_dropout_model.joblib"
SCALER_PATH   = MODELS_DIR / "scaler.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.joblib"
METRICS_PATH  = MODELS_DIR / "eval_metrics.joblib"


def train_and_save(X, y, feature_names: list):
    """
    Train RandomForest model with SMOTE for class imbalance.
    Saves model, scaler, feature names, and evaluation metrics.
    Returns evaluation metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Scale
    scaler = StandardScaler()
    X_res_scaled  = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    # RandomForest — robust, no native lib dependencies
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res_scaled, y_res)

    # Evaluation
    y_pred   = model.predict(X_test_scaled)
    y_prob   = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc  = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    cr       = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    # Feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    metrics = {
        "roc_auc": roc_auc,
        "average_precision": ap_score,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "fpr": fpr,
        "tpr": tpr,
        "precision": prec,
        "recall": rec,
        "feature_importance": fi_df,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    # Persist
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    joblib.dump(metrics,       METRICS_PATH)

    return metrics


def load_artifacts():
    """Load saved model, scaler and feature names. Returns (model, scaler, feature_names)."""
    if not MODEL_PATH.exists():
        return None, None, None
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


def load_metrics():
    """Load saved evaluation metrics dict."""
    if not METRICS_PATH.exists():
        return None
    return joblib.load(METRICS_PATH)


def predict_single(input_dict: dict, model, scaler, feature_names: list):
    """
    Predict dropout risk for a single student dict.
    Returns (probability, label, risk_tier).
    """
    row = np.array([input_dict.get(f, 0) for f in feature_names]).reshape(1, -1)
    row_scaled = scaler.transform(row)
    prob  = float(model.predict_proba(row_scaled)[0, 1])
    label = int(model.predict(row_scaled)[0])
    if prob >= 0.70:
        tier = "🔴 High Risk"
    elif prob >= 0.40:
        tier = "🟡 Moderate Risk"
    else:
        tier = "🟢 Low Risk"
    return prob, label, tier


def models_exist() -> bool:
    return MODEL_PATH.exists() and SCALER_PATH.exists()
