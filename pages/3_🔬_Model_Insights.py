"""
Page 3 — Model Insights & Evaluation Metrics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.model_utils import load_metrics, models_exist
from utils.charts import (
    roc_curve_plot, pr_curve_plot, confusion_matrix_plot,
    feature_importance_plot, histogram
)


@st.cache_data(show_spinner="Loading metrics…")
def get_metrics():
    return load_metrics()


def render():
    st.title("🔬 Model Insights & Evaluation")
    st.markdown(
        "Detailed evaluation of the **XGBoost Dropout Prediction Model** — "
        "trained on 80,000 students with SMOTE oversampling."
    )

    if not models_exist():
        st.warning(
            "⚠️ No trained model found. "
            "Please run **`python train_model.py`** from the project root first."
        )
        return

    metrics = get_metrics()
    if metrics is None:
        st.error("Metrics file not found. Retrain the model.")
        return

    # ── Performance KPIs ────────────────────────────────────────
    st.subheader("📊 Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Score",        f"{metrics['roc_auc']:.4f}")
    m2.metric("F1-Score",             f"{metrics['f1_score']:.4f}")
    m3.metric("Avg Precision (PR)",   f"{metrics['average_precision']:.4f}")
    cr = metrics["classification_report"]
    m4.metric("Accuracy",             f"{cr['accuracy']:.4f}")

    st.divider()

    # ── Classification Report ────────────────────────────────────
    st.subheader("📋 Classification Report")
    report_data = {
        "Class": ["Retained (0)", "Dropout (1)", "Macro Avg", "Weighted Avg"],
        "Precision": [
            cr["0"]["precision"], cr["1"]["precision"],
            cr["macro avg"]["precision"], cr["weighted avg"]["precision"]
        ],
        "Recall": [
            cr["0"]["recall"], cr["1"]["recall"],
            cr["macro avg"]["recall"], cr["weighted avg"]["recall"]
        ],
        "F1-Score": [
            cr["0"]["f1-score"], cr["1"]["f1-score"],
            cr["macro avg"]["f1-score"], cr["weighted avg"]["f1-score"]
        ],
        "Support": [
            int(cr["0"]["support"]), int(cr["1"]["support"]),
            int(cr["macro avg"]["support"]), int(cr["weighted avg"]["support"])
        ],
    }
    report_df = pd.DataFrame(report_data)
    report_df[["Precision","Recall","F1-Score"]] = report_df[["Precision","Recall","F1-Score"]].round(4)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── ROC + Confusion Matrix ───────────────────────────────────
    st.subheader("📈 ROC Curve & Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            roc_curve_plot(metrics["fpr"], metrics["tpr"], metrics["roc_auc"]),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            confusion_matrix_plot(metrics["confusion_matrix"]),
            use_container_width=True
        )

    # ── PR Curve ────────────────────────────────────────────────
    st.subheader("📉 Precision-Recall Curve")
    st.plotly_chart(
        pr_curve_plot(metrics["precision"], metrics["recall"], metrics["average_precision"]),
        use_container_width=True
    )

    st.divider()

    # ── Probability Distribution ─────────────────────────────────
    st.subheader("🎲 Predicted Probability Distribution")
    prob_df = pd.DataFrame({
        "probability": metrics["y_prob"],
        "actual": ["Dropout" if v == 1 else "Retained" for v in metrics["y_test"]]
    })
    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            prob_df, x="probability", color="actual", nbins=60,
            barmode="overlay", opacity=0.75, title="Dropout Probability — Test Set",
            color_discrete_map={"Retained": "#2ecc71", "Dropout": "#e74c3c"}
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        bins = pd.cut(prob_df["probability"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
        prob_df["risk_bin"] = bins
        bin_rates = prob_df.groupby("risk_bin")["actual"].apply(
            lambda x: (x == "Dropout").sum() / len(x) * 100
        ).reset_index(name="dropout_rate")
        fig2 = px.bar(bin_rates, x="risk_bin", y="dropout_rate",
                      title="Actual Dropout Rate by Predicted Probability Bin",
                      labels={"risk_bin": "Prob. Bin", "dropout_rate": "Actual Dropout Rate (%)"},
                      color="dropout_rate", color_continuous_scale="Reds")
        fig2.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Feature Importance ───────────────────────────────────────
    st.subheader("🏆 Feature Importance (XGBoost)")
    top_n = st.slider("Show top N features", 5, 35, 20)
    st.plotly_chart(
        feature_importance_plot(metrics["feature_importance"], top_n=top_n),
        use_container_width=True
    )

    # Top features table
    with st.expander("📄 Full Feature Importance Table"):
        fi = metrics["feature_importance"].copy()
        fi["importance"] = fi["importance"].round(6)
        st.dataframe(fi.reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── Model Info ───────────────────────────────────────────────
    st.subheader("ℹ️ Model Configuration")
    config_data = {
        "Parameter": [
            "Algorithm", "Estimators", "Max Depth", "Learning Rate",
            "Subsample", "ColSample ByTree", "Imbalance Handling",
            "Train/Test Split", "Random State"
        ],
        "Value": [
            "XGBoost (XGBClassifier)", "300", "6", "0.05",
            "0.8", "0.8", "SMOTE (imblearn)",
            "80% / 20% (stratified)", "42"
        ]
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
