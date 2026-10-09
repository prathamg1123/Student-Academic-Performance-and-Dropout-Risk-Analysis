"""
Page 2 — Student Dropout Risk Predictor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.model_utils import load_artifacts, predict_single, models_exist
from utils.data_loader import preprocess, load_raw_data, CATEGORICAL_COLS
from utils.charts import gauge_chart


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def _cat_maps():
    df_raw = load_raw_data()
    maps = {}
    for col in ["gender", "nationality", "family_income_level", "course_type", "study_mode"]:
        cats = sorted(df_raw[col].dropna().unique().tolist())
        maps[col] = {v: i for i, v in enumerate(cats)}
        maps[f"{col}_inv"] = cats
    return maps


def render():
    st.title("🎯 Student Dropout Risk Predictor")
    st.markdown(
        "Enter a student's profile below to estimate their **dropout probability** "
        "and receive a personalised risk tier assessment."
    )

    if not models_exist():
        st.warning(
            "⚠️ No trained model found. "
            "Please run **`python train_model.py`** from the project root first."
        )
        return

    model, scaler, feature_names = get_model()
    cat_maps = _cat_maps()

    # ── Input Form ───────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("🧑‍🎓 Student Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            age              = st.slider("Age", 17, 60, 21)
            gender_label     = st.selectbox("Gender",     cat_maps["gender_inv"])
            nationality_lbl  = st.selectbox("Nationality", cat_maps["nationality_inv"])
            first_gen        = st.selectbox("First Generation Student", ["No", "Yes"])
            disability       = st.selectbox("Disability Status", ["No", "Yes"])
            income_label     = st.selectbox("Family Income Level", cat_maps["family_income_level_inv"])

        with col2:
            scholarship      = st.selectbox("Scholarship Holder", ["No", "Yes"])
            tuition_ok       = st.selectbox("Tuition Up to Date", ["No", "Yes"])
            part_time        = st.selectbox("Part-Time Job", ["No", "Yes"])
            work_hours       = st.slider("Weekly Work Hours", 0, 60, 0)
            course_lbl       = st.selectbox("Course Type", cat_maps["course_type_inv"])
            mode_lbl         = st.selectbox("Study Mode",  cat_maps["study_mode_inv"])

        with col3:
            semester         = st.slider("Current Semester", 1, 10, 3)
            intl             = st.selectbox("International Student", ["No", "Yes"])
            admission_grade  = st.slider("Admission Grade",           50.0, 200.0, 130.0, 0.5)
            prev_qual_grade  = st.slider("Previous Qualification Grade", 50.0, 200.0, 130.0, 0.5)

        st.subheader("📚 Academic Performance")
        col4, col5, col6 = st.columns(3)
        with col4:
            units_enrolled   = st.slider("Units Enrolled",  1, 26, 12)
            units_passed     = st.slider("Units Passed",    0, 26, 10)
            units_failed     = st.slider("Units Failed",    0, 26,  2)
        with col5:
            gpa_s1           = st.slider("GPA Semester 1",  0.0, 4.0, 2.5, 0.01)
            gpa_s2           = st.slider("GPA Semester 2",  0.0, 4.0, 2.4, 0.01)
            attendance       = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
        with col6:
            assignments_sub  = st.slider("Assignments Submitted (%)", 0.0, 100.0, 80.0, 0.5)
            missed_deadlines = st.slider("Missed Deadlines",  0, 20, 2)
            extracurr        = st.selectbox("Extracurricular Activities", ["No", "Yes"])

        st.subheader("💡 Engagement & Wellbeing")
        col7, col8, col9 = st.columns(3)
        with col7:
            library_visits   = st.slider("Library Visits / Month",   0, 20, 3)
            lms_logins       = st.slider("LMS Logins / Week",        0.0, 30.0, 8.0, 0.1)
        with col8:
            stress_level     = st.slider("Stress Level (1-10)",      1, 10, 5)
            mental_health    = st.selectbox("Mental Health Support",  ["No", "Yes"])
        with col9:
            satisfaction     = st.slider("Satisfaction Score (1-10)", 1, 10, 6)
            commute_time     = st.slider("Commute Time (mins)",       0, 180, 30)
            sleep_hours      = st.slider("Sleep Hours / Night",       2.0, 12.0, 7.0, 0.5)

        submitted = st.form_submit_button("🔍 Predict Dropout Risk", use_container_width=True)

    # ── Prediction ───────────────────────────────────────────────
    if submitted:
        def yn(v): return 1 if v == "Yes" else 0

        # Compute engineered features (must match preprocess())
        pass_rate = units_passed / units_enrolled if units_enrolled > 0 else 0.0
        gpa_avg   = (gpa_s1 + gpa_s2) / 2
        gpa_change = gpa_s2 - gpa_s1
        engagement = (
            lms_logins * 0.3
            + library_visits * 0.2
            + attendance * 0.3
            + assignments_sub * 0.2
        )

        input_dict = {
            "age":                           age,
            "gender":                        cat_maps["gender"][gender_label],
            "nationality":                   cat_maps["nationality"][nationality_lbl],
            "first_generation":              yn(first_gen),
            "disability_status":             yn(disability),
            "family_income_level":           cat_maps["family_income_level"][income_label],
            "scholarship_holder":            yn(scholarship),
            "tuition_up_to_date":            yn(tuition_ok),
            "part_time_job":                 yn(part_time),
            "weekly_work_hours":             work_hours,
            "course_type":                   cat_maps["course_type"][course_lbl],
            "study_mode":                    cat_maps["study_mode"][mode_lbl],
            "semester":                      semester,
            "international_student":         yn(intl),
            "admission_grade":               admission_grade,
            "previous_qualification_grade":  prev_qual_grade,
            "units_enrolled":                units_enrolled,
            "units_passed":                  units_passed,
            "units_failed":                  units_failed,
            "gpa_semester_1":                gpa_s1,
            "gpa_semester_2":                gpa_s2,
            "attendance_rate":               attendance,
            "assignments_submitted_rate":    assignments_sub,
            "library_visits_per_month":      library_visits,
            "lms_logins_per_week":           lms_logins,
            "extracurricular_activities":    yn(extracurr),
            "missed_deadlines":              missed_deadlines,
            "stress_level":                  stress_level,
            "mental_health_support":         yn(mental_health),
            "satisfaction_score":            satisfaction,
            "commute_time_mins":             commute_time,
            "sleep_hours":                   sleep_hours,
            # engineered
            "pass_rate":                     pass_rate,
            "gpa_avg":                       gpa_avg,
            "gpa_change":                    gpa_change,
            "engagement_score":              engagement,
        }

        prob, label, tier = predict_single(input_dict, model, scaler, feature_names)

        st.divider()
        st.subheader("📋 Prediction Result")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with res_col2:
            st.markdown(f"### Risk Tier: {tier}")
            st.markdown(f"**Dropout Probability:** `{prob*100:.1f}%`")
            st.markdown(f"**Prediction Label:** `{'Dropout' if label == 1 else 'Retained'}`")

            # Personalised insights
            st.markdown("---")
            st.markdown("#### 💬 Key Risk Signals")
            signals = []
            if gpa_avg < 2.0:
                signals.append("🔴 GPA is critically low (below 2.0)")
            elif gpa_avg < 2.5:
                signals.append("🟡 GPA is below average (below 2.5)")
            if attendance < 60:
                signals.append("🔴 Attendance below 60% — high concern")
            elif attendance < 75:
                signals.append("🟡 Attendance is moderate — room to improve")
            if missed_deadlines >= 5:
                signals.append("🔴 High number of missed deadlines")
            if stress_level >= 8:
                signals.append("🔴 Very high stress level reported")
            if pass_rate < 0.5:
                signals.append("🔴 Pass rate below 50%")
            if tuition_ok == "No":
                signals.append("🟡 Tuition is not up to date")
            if gpa_change < -0.5:
                signals.append("🟡 GPA dropped significantly between semesters")
            if not signals:
                signals.append("🟢 No major risk signals detected")
            for s in signals:
                st.markdown(f"- {s}")

        # Recommendation card
        st.info(
            "**Recommendation:** "
            + (
                "Urgent academic counselling and support intervention recommended."
                if prob >= 0.70 else
                "Monitor student closely; consider proactive support."
                if prob >= 0.40 else
                "Student appears on track. Continue regular check-ins."
            )
        )

        # Summary table
        with st.expander("🗂️ Full Input Summary"):
            summary = pd.DataFrame.from_dict(input_dict, orient="index", columns=["Value"])
            st.dataframe(summary, use_container_width=True)
