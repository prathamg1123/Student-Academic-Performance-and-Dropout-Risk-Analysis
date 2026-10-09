"""
Page 1 — Overview & EDA Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_raw_data, preprocess, get_summary_stats
from utils.charts import (
    pie_chart, bar_chart, histogram, box_plot,
    scatter_plot, correlation_heatmap
)


@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    df_raw = load_raw_data()
    df = preprocess(df_raw)
    df["dropout_label"] = df["dropout"].map({0: "Retained", 1: "Dropout"})
    return df_raw, df


def render():
    st.title("📊 Student Academic Performance — Overview & EDA")
    st.markdown("Explore the dataset of **100,000 students** across demographics, academics, and engagement.")

    df_raw, df = get_data()
    stats = get_summary_stats(df)

    # ── KPI Row ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",   f"{stats['total_students']:,}")
    c2.metric("Dropout Count",    f"{stats['dropout_count']:,}")
    c3.metric("Dropout Rate",     f"{stats['dropout_rate']}%")
    c4.metric("Avg GPA",          f"{stats['avg_gpa']}")
    c5.metric("Avg Attendance",   f"{stats['avg_attendance']}%")

    c1b, c2b, c3b = st.columns(3)
    c1b.metric("Avg Stress Level",    f"{stats['avg_stress']} / 10")
    c2b.metric("Avg Satisfaction",    f"{stats['avg_satisfaction']} / 10")
    c3b.metric("Retained Students",   f"{stats['retained_count']:,}")

    st.divider()

    # ── Dropout Distribution ────────────────────────────────────
    st.subheader("🎯 Dropout Distribution")
    col1, col2 = st.columns(2)
    with col1:
        labels = ["Retained", "Dropout"]
        values = [stats["retained_count"], stats["dropout_count"]]
        st.plotly_chart(pie_chart(labels, values, "Overall Dropout vs Retained"),
                        use_container_width=True)
    with col2:
        grp = df.groupby(["course_type", "dropout_label"]).size().reset_index(name="count")
        # course_type is encoded; re-map from raw
        course_map = dict(enumerate(pd.Categorical(df_raw["course_type"]).categories))
        grp["course"] = grp["course_type"].map(course_map)
        st.plotly_chart(
            bar_chart(grp, x="course", y="count", color="dropout_label",
                      title="Dropout by Course Type", barmode="group"),
            use_container_width=True
        )

    st.divider()

    # ── Demographics ────────────────────────────────────────────
    st.subheader("👥 Demographics & Background")
    col3, col4 = st.columns(2)
    with col3:
        gender_grp = df_raw.groupby(["gender", df["dropout_label"]]).size().reset_index(name="count")
        gender_grp.columns = ["gender", "status", "count"]
        st.plotly_chart(
            bar_chart(gender_grp, x="gender", y="count", color="status",
                      title="Dropout by Gender", barmode="group"),
            use_container_width=True
        )
    with col4:
        income_grp = df_raw.groupby(["family_income_level", df["dropout_label"]]).size().reset_index(name="count")
        income_grp.columns = ["income", "status", "count"]
        order = {"Low": 0, "Middle": 1, "High": 2}
        income_grp["_order"] = income_grp["income"].map(order)
        income_grp = income_grp.sort_values("_order")
        st.plotly_chart(
            bar_chart(income_grp, x="income", y="count", color="status",
                      title="Dropout by Family Income Level", barmode="group"),
            use_container_width=True
        )

    col5, col6 = st.columns(2)
    with col5:
        mode_grp = df_raw.groupby(["study_mode", df["dropout_label"]]).size().reset_index(name="count")
        mode_grp.columns = ["mode", "status", "count"]
        st.plotly_chart(
            bar_chart(mode_grp, x="mode", y="count", color="status",
                      title="Dropout by Study Mode", barmode="group"),
            use_container_width=True
        )
    with col6:
        schol_grp = df_raw.copy()
        schol_grp["dropout_label"] = df["dropout_label"]
        schol_grp["Scholarship"] = schol_grp["scholarship_holder"].map({1: "Yes", 0: "No"})
        grp2 = schol_grp.groupby(["Scholarship", "dropout_label"]).size().reset_index(name="count")
        st.plotly_chart(
            bar_chart(grp2, x="Scholarship", y="count", color="dropout_label",
                      title="Dropout by Scholarship Status", barmode="group"),
            use_container_width=True
        )

    st.divider()

    # ── Academic Performance ─────────────────────────────────────
    st.subheader("📚 Academic Performance Analysis")
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(
            histogram(df, x="gpa_avg", color="dropout_label",
                      title="GPA Distribution by Dropout Status", nbins=50),
            use_container_width=True
        )
    with col8:
        st.plotly_chart(
            box_plot(df, x="dropout_label", y="gpa_avg",
                     color="dropout_label", title="GPA vs Dropout Status"),
            use_container_width=True
        )

    col9, col10 = st.columns(2)
    with col9:
        st.plotly_chart(
            histogram(df, x="attendance_rate", color="dropout_label",
                      title="Attendance Rate by Dropout Status", nbins=40),
            use_container_width=True
        )
    with col10:
        st.plotly_chart(
            scatter_plot(df.sample(3000, random_state=42),
                         x="gpa_avg", y="attendance_rate",
                         color="dropout_label",
                         title="GPA vs Attendance (sample 3k)"),
            use_container_width=True
        )

    st.divider()

    # ── Engagement & Wellbeing ───────────────────────────────────
    st.subheader("💡 Engagement & Wellbeing")
    col11, col12 = st.columns(2)
    with col11:
        st.plotly_chart(
            box_plot(df, x="dropout_label", y="stress_level",
                     color="dropout_label", title="Stress Level vs Dropout"),
            use_container_width=True
        )
    with col12:
        st.plotly_chart(
            box_plot(df, x="dropout_label", y="engagement_score",
                     color="dropout_label", title="Engagement Score vs Dropout"),
            use_container_width=True
        )

    col13, col14 = st.columns(2)
    with col13:
        st.plotly_chart(
            box_plot(df, x="dropout_label", y="satisfaction_score",
                     color="dropout_label", title="Satisfaction Score vs Dropout"),
            use_container_width=True
        )
    with col14:
        st.plotly_chart(
            histogram(df, x="missed_deadlines", color="dropout_label",
                      title="Missed Deadlines by Dropout Status", nbins=20),
            use_container_width=True
        )

    st.divider()

    # ── Correlation Matrix ───────────────────────────────────────
    st.subheader("🔗 Feature Correlation Matrix")
    numeric_features = [
        "age", "admission_grade", "gpa_semester_1", "gpa_semester_2",
        "gpa_avg", "gpa_change", "attendance_rate", "assignments_submitted_rate",
        "stress_level", "satisfaction_score", "engagement_score",
        "missed_deadlines", "pass_rate", "sleep_hours", "dropout"
    ]
    st.plotly_chart(
        correlation_heatmap(df, numeric_features),
        use_container_width=True
    )

    st.divider()

    # ── Raw Data Preview ─────────────────────────────────────────
    with st.expander("🗂️ Raw Dataset Preview (first 500 rows)"):
        st.dataframe(df_raw.head(500), use_container_width=True)
        st.caption(f"Full dataset: {len(df_raw):,} rows × {df_raw.shape[1]} columns")
