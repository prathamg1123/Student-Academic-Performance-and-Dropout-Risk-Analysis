"""Shared chart/plot helpers using Plotly."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PALETTE = px.colors.qualitative.Set2
DROPOUT_COLORS = {"Retained": "#2ecc71", "Dropout": "#e74c3c"}


def pie_chart(labels, values, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=[DROPOUT_COLORS.get(l, "#3498db") for l in labels],
        hole=0.45, textinfo="percent+label"
    ))
    fig.update_layout(title_text=title, showlegend=True, height=340,
                      margin=dict(t=50, b=10, l=10, r=10))
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = "",
              orientation="v", barmode="group"):
    fig = px.bar(df, x=x, y=y, color=color, title=title, barmode=barmode,
                 orientation=orientation,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=380, margin=dict(t=50, b=30, l=30, r=10))
    return fig


def histogram(df: pd.DataFrame, x: str, color: str = None, title: str = "",
              nbins: int = 40, marginal: str = None):
    fig = px.histogram(df, x=x, color=color, title=title, nbins=nbins,
                       marginal=marginal, barmode="overlay", opacity=0.75,
                       color_discrete_sequence=PALETTE)
    fig.update_layout(height=380, margin=dict(t=50, b=30, l=30, r=10))
    return fig


def box_plot(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = ""):
    fig = px.box(df, x=x, y=y, color=color, title=title,
                 color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(height=380, margin=dict(t=50, b=30, l=30, r=10))
    return fig


def scatter_plot(df: pd.DataFrame, x: str, y: str, color: str = None,
                 size: str = None, title: str = ""):
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title,
                     color_discrete_sequence=PALETTE, opacity=0.6)
    fig.update_layout(height=420, margin=dict(t=50, b=30, l=30, r=10))
    return fig


def roc_curve_plot(fpr, tpr, auc_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})",
                             line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", height=400,
                      margin=dict(t=50, b=40, l=40, r=10))
    return fig


def pr_curve_plot(precision, recall, ap_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name=f"PR (AP={ap_val:.3f})",
                             line=dict(color="#e67e22", width=2)))
    fig.update_layout(title="Precision-Recall Curve",
                      xaxis_title="Recall", yaxis_title="Precision",
                      height=400, margin=dict(t=50, b=40, l=40, r=10))
    return fig


def confusion_matrix_plot(cm: np.ndarray):
    labels = ["Retained", "Dropout"]
    z_text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=z_text, texttemplate="%{text}",
        colorscale="Blues", showscale=False
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=380, margin=dict(t=50, b=40, l=60, r=10)
    )
    return fig


def feature_importance_plot(fi_df: pd.DataFrame, top_n: int = 20):
    df = fi_df.head(top_n).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 title=f"Top {top_n} Feature Importances (XGBoost)",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(height=520, margin=dict(t=50, b=30, l=10, r=10),
                      coloraxis_showscale=False)
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list):
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(title="Correlation Matrix", height=600,
                      margin=dict(t=50, b=10, l=10, r=10))
    return fig


def gauge_chart(probability: float, title: str = "Dropout Risk Score"):
    color = "#e74c3c" if probability >= 0.70 else ("#f39c12" if probability >= 0.40 else "#2ecc71")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        title={"text": title, "font": {"size": 18}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#d5f5e3"},
                {"range": [40, 70], "color": "#fef9e7"},
                {"range": [70, 100],"color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
    return fig
