"""
Main Streamlit entry point — Home / Landing page.
Run:  streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar global info ──────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=80
    )
    st.markdown("## 🎓 Student Dropout\n### Risk Prediction")
    st.divider()
    st.markdown(
        """
**Navigate using the pages above:**

📊 **Overview & EDA** — Dataset exploration, charts, and statistics

🎯 **Prediction** — Input a student profile and get a live risk score

🔬 **Model Insights** — ROC, PR curve, confusion matrix & feature importance
        """
    )
    st.divider()
    st.caption("Dataset: `student_dropout_risk_2026.csv`  \n100,000 students · 33 features")
    st.caption("Model: XGBoost + SMOTE  \nBuilt with Streamlit & Plotly")

# ── Home content ─────────────────────────────────────────────────────────────
st.title("🎓 Student Academic Performance Analysis")
st.subheader("& Dropout Risk Prediction System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
### 🔍 About This Project

This interactive data analysis and machine learning application helps educators and
administrators **identify students at risk of dropping out** before it happens —
enabling timely interventions and personalised support.

Built on a dataset of **100,000 students** with **33 features** spanning academic
performance, demographics, engagement, financial status, and wellbeing.

---

### 🚀 Quick Start

| Step | Action |
|------|--------|
| 1️⃣ | **Train the model** — run `python train_model.py` once |
| 2️⃣ | Navigate to **📊 Overview & EDA** to explore the dataset |
| 3️⃣ | Visit **🎯 Prediction** to test individual student risk scoring |
| 4️⃣ | Check **🔬 Model Insights** for evaluation metrics and feature importance |

---

### 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend  | Streamlit, Plotly Express, Plotly Go |
| ML Model  | XGBoost, Scikit-learn, imbalanced-learn (SMOTE) |
| Data      | Pandas, NumPy |
| Persistence | Joblib |

        """
    )

with col2:
    st.markdown("### 📊 Dataset Overview")
    kpis = [
        ("Total Records",   "100,000"),
        ("Features",        "33"),
        ("Target",          "Dropout (Binary)"),
        ("Model",           "XGBoost"),
        ("Imbalance Fix",   "SMOTE"),
        ("Train/Test",      "80% / 20%"),
    ]
    for k, v in kpis:
        st.metric(k, v)

st.divider()

# Feature categories
st.markdown("### 🗂️ Feature Categories")
cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
with cat_col1:
    st.markdown("""
**👤 Demographics**
- Age, Gender, Nationality
- First Generation
- Disability Status
- International Student
""")
with cat_col2:
    st.markdown("""
**💰 Financial**
- Family Income Level
- Scholarship Holder
- Tuition Up to Date
- Part-Time Job
- Weekly Work Hours
""")
with cat_col3:
    st.markdown("""
**📚 Academic**
- Admission Grade
- GPA Semester 1 & 2
- Units Enrolled/Passed/Failed
- Attendance Rate
- Assignments Submitted
""")
with cat_col4:
    st.markdown("""
**💡 Wellbeing**
- Stress Level
- Satisfaction Score
- Sleep Hours
- Mental Health Support
- LMS Logins / Week
- Missed Deadlines
""")
