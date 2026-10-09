# 🎓 Student Academic Performance Analysis & Dropout Risk Prediction

A full-stack data analysis and machine learning project built with **Python**, **Streamlit**, **XGBoost**, and **Plotly**.

---

## 📁 Project Structure

```
student_dropout_project/
├── app.py                          # Streamlit entry point (Home page)
├── train_model.py                  # One-time model training script
├── requirements.txt                # Python dependencies
├── data/
│   └── student_dropout_risk_2026.csv
├── models/                         # Saved model artifacts (after training)
│   ├── xgb_dropout_model.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   └── eval_metrics.joblib
├── pages/
│   ├── 1_📊_Overview_EDA.py        # Dataset exploration & EDA
│   ├── 2_🎯_Prediction.py          # Live dropout risk prediction
│   └── 3_🔬_Model_Insights.py      # Model evaluation metrics
├── utils/
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── model_utils.py              # Model train/predict/persist
│   └── charts.py                  # Shared Plotly chart helpers
└── .streamlit/
    └── config.toml                 # Theme & server config
```

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
cd student_dropout_project
pip install -r requirements.txt
```

### 2. Train the model (once)

```bash
python train_model.py
```

Expected output:
```
[1/3] Loading dataset...      Loaded 100,000 records
[2/3] Preprocessing...        Features: 36 | Dropout rate: ~32%
[3/3] Training XGBoost ...
✅  Training complete!
    ROC-AUC   : ~0.87+
    F1-Score  : ~0.79+
    Avg Prec  : ~0.83+
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Features

### Page 1 — Overview & EDA
- KPI cards (total students, dropout rate, avg GPA, attendance, stress)
- Dropout distribution: pie chart + by course type
- Demographics: dropout by gender, income, study mode, scholarship
- Academic performance: GPA histograms, box plots, scatter plots
- Engagement & wellbeing: stress, satisfaction, missed deadlines
- Correlation heatmap of 15 key features
- Raw data preview

### Page 2 — Dropout Risk Predictor
- 30+ input controls for full student profile entry
- Real-time XGBoost prediction after form submission
- Gauge chart with probability (0–100%)
- Risk tier: 🟢 Low / 🟡 Moderate / 🔴 High
- Personalised risk signal explanations
- Recommendation card for educators

### Page 3 — Model Insights
- Performance KPIs (ROC-AUC, F1, Avg Precision, Accuracy)
- Full classification report table
- ROC curve + Confusion matrix side-by-side
- Precision-Recall curve
- Predicted probability distribution histograms
- Calibration: actual dropout rate by predicted probability bin
- Top-N feature importance bar chart (adjustable slider)
- Full feature importance table
- Model configuration summary

---

## 🗂️ Dataset: `student_dropout_risk_2026.csv`

| Attribute | Detail |
|-----------|--------|
| Rows      | 100,000 |
| Features  | 33 |
| Target    | `dropout` (0 = Retained, 1 = Dropout) |
| Source    | Synthetic academic dataset |

**Feature categories:** Demographics, Financial, Academic Performance, Engagement, Wellbeing

---

## 🤖 Model Details

| Setting | Value |
|---------|-------|
| Algorithm | XGBoost (XGBClassifier) |
| Estimators | 300 |
| Max Depth | 6 |
| Learning Rate | 0.05 |
| Imbalance | SMOTE (imblearn) |
| Split | 80/20 stratified |
| Scaling | StandardScaler |

---

## 📦 Tech Stack

| Layer | Library |
|-------|---------|
| Frontend | Streamlit, Plotly Express, Plotly Graph Objects |
| ML | XGBoost, Scikit-learn |
| Imbalance | imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Persistence | Joblib |
