"""
Diabetes Prediction Dashboard  ·  Redesigned
Senior Data Scientist / Front-end refactor
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Intelligence Platform",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Main area ── */
.main .block-container { padding-top: 2rem; max-width: 1200px; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid #1e40af33;
}
.page-header h1 { color: #f0f9ff; font-size: 1.9rem; font-weight: 700; margin: 0; }
.page-header p  { color: #93c5fd; font-size: 0.95rem; margin: 0.4rem 0 0; }
.page-header-inner {
    display: flex;
    align-items: center;
    gap: 1.25rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-card .label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { color: #f1f5f9; font-size: 2rem; font-weight: 700; line-height: 1.2; }
.metric-card .delta { font-size: 0.82rem; margin-top: 0.2rem; }
.delta-pos { color: #34d399; }
.delta-neg { color: #f87171; }

/* ── Risk result cards ── */
.risk-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.risk-high h2 { color: #fca5a5; font-size: 1.5rem; margin: 0; }
.risk-high p  { color: #fecaca; margin: 0.5rem 0 0; font-size: 0.9rem; }

.risk-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.risk-low h2 { color: #86efac; font-size: 1.5rem; margin: 0; }
.risk-low p  { color: #bbf7d0; margin: 0.5rem 0 0; font-size: 0.9rem; }

/* ── Section label ── */
.section-label {
    color: #3b82f6;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.section-title {
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.2rem;
}

/* ── Info box ── */
.info-box {
    background: #0f2137;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #bfdbfe;
}

/* ── Factor badge ── */
.factor-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.9rem;
    background: #1e293b;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.factor-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* ── Gauge container ── */
.gauge-wrap {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #fff !important;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

/* ── Plotly transparent bg ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Chatbot styles ── */
.chat-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid #1e40af33;
}
.chat-header h1 { color: #f0f9ff; font-size: 1.9rem; font-weight: 700; margin: 0; }
.chat-header p  { color: #93c5fd; font-size: 0.95rem; margin: 0.4rem 0 0; }
.chat-header .page-header-inner { display: flex; align-items: center; gap: 1.25rem; }

.chat-bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #f0f9ff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.15rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px #1d4ed833;
}
.chat-bubble-assistant {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.15rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.55;
}
.chat-role-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.2rem;
}
.chat-role-user { color: #93c5fd; text-align: right; margin-right: 0.3rem; }
.chat-role-assistant { color: #64748b; margin-left: 0.3rem; }
.chat-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.chat-empty {
    text-align: center;
    padding: 1.5rem 1rem;
    color: #475569;
}
.chat-empty .big-icon { display: none; }
.chat-empty p { font-size: 0.9rem; color: #64748b; line-height: 1.6; }
.chat-suggestion {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.83rem;
    color: #93c5fd;
    cursor: pointer;
    transition: border-color 0.2s;
    margin-bottom: 0.5rem;
}
.api-key-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}


/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 1.2rem 0.5rem 1.6rem;
    border-bottom: 1px solid #334155;
    margin-bottom: 1.4rem;
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.brand-text {
    display: flex;
    flex-direction: column;
}
.brand-name {
    color: #f0f9ff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    line-height: 1.2;
}
.brand-sub {
    color: #64748b !important;
    font-size: 0.7rem !important;
    margin-top: 0.1rem;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.65rem 1rem;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px #1d4ed844;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    box-shadow: 0 6px 18px #1d4ed866;
    transform: translateY(-1px);
}

/* ── Summary insight card ── */
.insight-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.9rem;
}
.insight-card h4 { color: #93c5fd; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem; }
.insight-card p  { color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme defaults ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=10, r=10, t=40, b=10),
)
PALETTE     = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]
RISK_COLOR  = ["#22c55e", "#ef4444"]

# ── Data & Model caching ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "diabetes_prediction_dataset_cleaned.csv")
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base, "xgboost_model.pkl"))

@st.cache_data(show_spinner=False)
def precompute_analytics(df):
    """Pre-compute all heavy aggregations once so tab switches are instant."""
    diabetic     = df[df["diabetes"] == 1]
    non_diabetic = df[df["diabetes"] == 0]

    # Violin samples — reduced for faster rendering, still statistically representative
    sample_nd = non_diabetic.sample(min(3000, len(non_diabetic)), random_state=1)
    sample_d  = diabetic.sample(min(3000, len(diabetic)), random_state=1)

    # Scatter & bubble samples
    scatter_sample = df.sample(min(2500, len(df)), random_state=42)
    bubble_sample  = df.sample(min(1500, len(df)), random_state=99)

    # Categorical aggregations
    cat_col_list = ["gender", "smoking_history", "age_group", "bmi_cat"]
    cat_aggs = {}
    for col in cat_col_list:
        grp = df.groupby(col)["diabetes"].agg(["mean", "sum", "count"]).reset_index()
        grp.columns = [col, "rate", "diabetic_count", "total"]
        grp["rate_pct"] = (grp["rate"] * 100).round(2)
        cat_aggs[col] = grp.sort_values("rate_pct", ascending=True)

    # Pie value counts
    pie_counts = {col: df[col].value_counts() for col in cat_col_list}

    # Comorbidity aggregations
    comorbidity_aggs = {}
    for col in ["hypertension", "heart_disease"]:
        comorbidity_aggs[col] = df.groupby(col)["diabetes"].mean().reset_index()

    # Correlation matrix
    corr_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                 "hypertension", "heart_disease", "diabetes"]
    corr = df[corr_cols].corr().round(3)

    # Descriptive stats
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    stats = {
        "all":          df[num_cols].describe().T.round(2),
        "diabetic":     diabetic[num_cols].describe().T.round(2),
        "non_diabetic": non_diabetic[num_cols].describe().T.round(2),
    }

    return dict(
        diabetic=diabetic, non_diabetic=non_diabetic,
        sample_nd=sample_nd, sample_d=sample_d,
        scatter_sample=scatter_sample, bubble_sample=bubble_sample,
        cat_col_list=cat_col_list, cat_aggs=cat_aggs, pie_counts=pie_counts,
        comorbidity_aggs=comorbidity_aggs,
        corr=corr, corr_cols=corr_cols,
        stats=stats, num_cols=num_cols,
    )

# ── Feature engineering helpers ───────────────────────────────────────────────
def categorize_age(age):
    if age < 12:   return "Child"
    if age < 18:   return "Teen"
    if age < 35:   return "Young Adult"
    if age < 60:   return "Middle Aged"
    return "Senior"

def categorize_bmi(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Healthy"
    if bmi < 30:   return "Overweight"
    if bmi < 40:   return "Obese"
    return "Extremely Obese"

def prepare_input(gender, age, hypertension, heart_disease,
                  smoking_history, bmi, hba1c, glucose):
    return pd.DataFrame([{
        "gender":              gender,
        "age":                 age,
        "hypertension":        hypertension,
        "heart_disease":       heart_disease,
        "smoking_history":     smoking_history,
        "bmi":                 bmi,
        "HbA1c_level":         hba1c,
        "blood_glucose_level": glucose,
        "age_group":           categorize_age(age),
        "bmi_cat":             categorize_bmi(bmi),
    }])

def risk_level(prob):
    if prob < 0.20: return "Low",      "#22c55e"
    if prob < 0.50: return "Moderate", "#f59e0b"
    if prob < 0.75: return "High",     "#f97316"
    return "Very High",                "#ef4444"

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-logo">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="38" height="38" rx="10" fill="#1d4ed8"/>
                <!-- cross / plus symbol -->
                <rect x="16" y="8" width="6" height="22" rx="3" fill="white"/>
                <rect x="8" y="16" width="22" height="6" rx="3" fill="white"/>
                <!-- small circle accent -->
                <circle cx="28" cy="10" r="4" fill="#60a5fa"/>
            </svg>
            <div class="brand-text">
                <span class="brand-name">DiabetesIQ</span>
                <span class="brand-sub">Clinical Intelligence Platform</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Prediction", "Data Analysis", "AI Assistant"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#1e293b; margin: 1rem 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "Prediction":

    # ── Sidebar inputs ──
    with st.sidebar:
        st.markdown("<div class='section-label'>Patient Profile</div>", unsafe_allow_html=True)
        gender  = st.selectbox("Gender", ["Female", "Male"])
        age     = st.slider("Age", 0, 80, 45)
        bmi     = st.number_input("BMI", 10.0, 70.0, 27.0, step=0.1, format="%.1f")
        hba1c   = st.number_input("HbA1c Level (%)", 3.5, 9.0, 5.5, step=0.1, format="%.1f")
        glucose = st.number_input("Blood Glucose (mg/dL)", 80, 300, 130)

        st.markdown("<div class='section-label' style='margin-top:1rem'>Medical History</div>", unsafe_allow_html=True)
        hypertension  = st.selectbox("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x else "No")
        heart_disease = st.selectbox("Heart Disease",  [0, 1], format_func=lambda x: "Yes" if x else "No")
        smoking       = st.selectbox("Smoking History",
                                     ["never", "former", "current", "not current", "ever", "No Info"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", use_container_width=True)

    # ── Page header ──
    st.markdown("""
    <div class="page-header">
        <div class="page-header-inner">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
                <rect width="48" height="48" rx="14" fill="#1d4ed822"/>
                <path d="M24 10 L24 38 M10 24 L38 24" stroke="#60a5fa" stroke-width="5" stroke-linecap="round"/>
                <circle cx="34" cy="14" r="5" fill="#3b82f6" opacity="0.8"/>
            </svg>
            <div>
                <h1>Diabetes Risk Prediction</h1>
                <p>Enter patient data in the sidebar and run the XGBoost model to get an instant risk assessment.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Patient snapshot KPIs ──
    bmi_cat = categorize_bmi(bmi)
    age_grp = categorize_age(age)

    bmi_color  = "#22c55e" if bmi_cat == "Healthy" else ("#f59e0b" if bmi_cat == "Overweight" else "#ef4444")
    hba1c_flag = "Pre-diabetic range" if hba1c >= 5.7 else "Normal range"
    glc_flag   = "Elevated" if glucose >= 140 else "Normal"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Age / Group</div>
            <div class="value">{int(age)}</div>
            <div class="delta">{age_grp}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">BMI</div>
            <div class="value" style="color:{bmi_color}">{bmi:.1f}</div>
            <div class="delta">{bmi_cat}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">HbA1c Level</div>
            <div class="value">{hba1c:.1f}<span style="font-size:1rem">%</span></div>
            <div class="delta">{hba1c_flag}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Blood Glucose</div>
            <div class="value">{glucose}</div>
            <div class="delta">{glc_flag}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Prediction output ──
    if predict_btn:
        model = load_model()
        X     = prepare_input(gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose)
        prob  = float(model.predict_proba(X)[0][1])
        THRESHOLD = 0.3
        #pred  = int(model.predict(X)[0])
        pred = 1 if prob >= THRESHOLD else 0
        rlabel, rcolor = risk_level(prob)

        left, right = st.columns([1, 1], gap="large")

        with left:
            # Risk verdict
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>Diabetic — High Risk</h2>
                    <p>This patient profile indicates a <b>{rlabel}</b> diabetes risk level.<br>
                    Clinical review and confirmatory testing are strongly recommended.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>Non-Diabetic — Low Risk</h2>
                    <p>This patient profile indicates a <b>{rlabel}</b> diabetes risk level.<br>
                    Continue routine screening based on age and clinical history.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk factor breakdown
            st.markdown("<div class='section-label'>Key Risk Factors Detected</div>", unsafe_allow_html=True)
            factors = []
            if hba1c >= 6.5:   factors.append(("critical", "HbA1c ≥ 6.5% — Diabetic threshold"))
            elif hba1c >= 5.7: factors.append(("warning", "HbA1c 5.7–6.4% — Pre-diabetic range"))
            if glucose >= 200:  factors.append(("critical", f"Blood glucose {glucose} mg/dL — Severely elevated"))
            elif glucose >= 140:factors.append(("warning", f"Blood glucose {glucose} mg/dL — Elevated"))
            if bmi >= 30:       factors.append(("moderate", f"BMI {bmi:.1f} — {bmi_cat}"))
            if hypertension:    factors.append(("moderate", "Hypertension present"))
            if heart_disease:   factors.append(("moderate", "Heart disease present"))
            if smoking == "current": factors.append(("warning", "Current smoker"))

            if factors:
                for icon, text in factors:
                    st.markdown(f"""
                    <div class="factor-row">
                        <span class="factor-dot" style="background:{'#ef4444' if icon=='critical' else '#f59e0b' if icon=='warning' else '#f97316'}"></span>
                        <span>{text}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="factor-row">
                    <span class="factor-dot" style="background:#22c55e"></span>
                    <span>No major risk flags detected in this profile.</span>
                </div>""", unsafe_allow_html=True)

        with right:
            # Gauge chart
            gauge_color = rcolor
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                title = dict(text="Diabetes Probability", font=dict(size=16, color="#94a3b8")),
                number= dict(suffix="%", font=dict(size=42, color="#f1f5f9")),
                delta = dict(reference=8.8, valueformat=".1f",
                             increasing=dict(color="#ef4444"),
                             decreasing=dict(color="#22c55e")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#334155",
                              tickfont=dict(color="#64748b")),
                    bar=dict(color=gauge_color, thickness=0.28),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 20],  color="rgba(5, 46, 22, 0.9)"),
                        dict(range=[20, 50], color="rgba(20, 83, 45, 0.2)"),
                        dict(range=[50, 75], color="rgba(120, 53, 15, 0.2)"),
                        dict(range=[75, 100],color="rgba(69, 10, 10, 0.33)"),
                    ],
                    threshold=dict(line=dict(color="#f1f5f9", width=3), value=50),
                )
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.markdown("<div class='gauge-wrap'>", unsafe_allow_html=True)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            # Risk scale legend
            st.markdown("""
            <div class="info-box" style="margin-top:1rem">
                <b style="color:#93c5fd">Risk Scale</b><br>
                <span style="color:#22c55e">&#9632;</span> <b>Low</b> &lt; 20% &nbsp;&middot;&nbsp;
                <span style="color:#f59e0b">&#9632;</span> <b>Moderate</b> 20–50% &nbsp;&middot;&nbsp;
                <span style="color:#f97316">&#9632;</span> <b>High</b> 50–75% &nbsp;&middot;&nbsp;
                <span style="color:#ef4444">&#9632;</span> <b>Very High</b> &gt; 75%<br><br>
                <span style="color:#64748b; font-size:0.8rem">
                This tool is intended for clinical decision support only and does not replace professional diagnosis.
                </span>
            </div>""", unsafe_allow_html=True)

    else:
        # Idle state — show instructions
        st.markdown("""
        <div class="info-box">
            <b>Fill in patient details</b> in the sidebar and click <b>Run Prediction</b> to generate
            a risk assessment powered by an XGBoost classifier trained on 96,000+ patient records.
        </div>""", unsafe_allow_html=True)

        # Model specs card
        st.markdown("<div class='section-label'>Model Overview</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        specs = [
            ("Algorithm",     "XGBoost",   "Gradient boosted decision trees"),
            ("Training Set",  "96,128",    "Cleaned & feature-engineered records"),
            ("Features",      "10",        "Including engineered age & BMI groups"),
            ("Prevalence",    "8.8%",      "Diabetic patients in training data"),
        ]
        for col, (title, val, sub) in zip([m1, m2, m3, m4], specs):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{title}</div>
                    <div class="value" style="font-size:1.4rem">{val}</div>
                    <div class="delta" style="color:#64748b">{sub}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analysis":
    # ── Page header ──
    st.markdown("""
    <div class="page-header">
        <div class="page-header-inner">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
                <rect width="48" height="48" rx="14" fill="#1d4ed822"/>
                <rect x="10" y="28" width="6" height="12" rx="2" fill="#3b82f6"/>
                <rect x="20" y="20" width="6" height="20" rx="2" fill="#60a5fa"/>
                <rect x="30" y="12" width="6" height="28" rx="2" fill="#93c5fd"/>
                <polyline points="10,24 20,16 30,20 40,10" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
            <div>
                <h1>Data Analysis & Insights</h1>
                <p>Exploratory analysis of 96,128 patient records — distributions, correlations, and diabetes risk patterns.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    df  = load_data()
    a   = precompute_analytics(df)   # all heavy work cached after first run
    diabetic     = a["diabetic"]
    non_diabetic = a["non_diabetic"]
    n_total = len(df)
    n_pos   = diabetic.shape[0]
    n_neg   = non_diabetic.shape[0]

    # ── Top KPIs ──
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        ("Total Records",     f"{n_total:,}",   "Full dataset"),
        ("Diabetic",          f"{n_pos:,}",     f"{n_pos/n_total*100:.1f}% prevalence"),
        ("Non-Diabetic",      f"{n_neg:,}",     f"{n_neg/n_total*100:.1f}% of records"),
        ("Mean HbA1c",        f"{df['HbA1c_level'].mean():.2f}%", "Normal < 5.7%"),
        ("Mean Glucose",      f"{df['blood_glucose_level'].mean():.0f}",  "mg/dL"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4, k5], kpi_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.5rem">{val}</div>
                <div class="delta" style="color:#64748b">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab_num, tab_cat, tab_corr, tab_summary = st.tabs([
        "Numerical Analysis",
        "Categorical Analysis",
        "Correlations",
        "Summary & Conclusions",
    ])

    # ─────────────────────────────────────────────────
    # TAB 1 — Numerical Analysis
    # ─────────────────────────────────────────────────
    with tab_num:
        st.markdown("<br>", unsafe_allow_html=True)
        num_cols = a["num_cols"]
        nice     = {"age": "Age", "bmi": "BMI",
                    "HbA1c_level": "HbA1c Level (%)", "blood_glucose_level": "Blood Glucose (mg/dL)"}

        # Distribution — 2x2 subplot grid of vertical violin+box plots
        st.markdown("<div class='section-label'>Feature Distributions by Diabetes Status</div>", unsafe_allow_html=True)

        fig_violins = make_subplots(
            rows=2, cols=2,
            subplot_titles=[nice[c] for c in num_cols],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        sample_nd = a["sample_nd"]
        sample_d  = a["sample_d"]

        positions = [(1,1),(1,2),(2,1),(2,2)]
        show_legend_flags = [True, False, False, False]

        for idx, col in enumerate(num_cols):
            r, c = positions[idx]
            show = show_legend_flags[idx]

            fig_violins.add_trace(go.Violin(
                y=sample_nd[col],
                name="Non-Diabetic",
                legendgroup="Non-Diabetic",
                showlegend=show,
                line_color="#3b82f6",
                fillcolor="rgba(59, 130, 246, 0.18)",
                box_visible=True,
                meanline_visible=True,
                points=False,
                side="negative",
                width=1.8,
            ), row=r, col=c)

            fig_violins.add_trace(go.Violin(
                y=sample_d[col],
                name="Diabetic",
                legendgroup="Diabetic",
                showlegend=show,
                line_color="#ef4444",
                fillcolor="rgba(239, 68, 68, 0.18)",
                box_visible=True,
                meanline_visible=True,
                points=False,
                side="positive",
                width=1.8,
            ), row=r, col=c)

        fig_violins.update_layout(
            **PLOTLY_LAYOUT,
            height=600,
            violingap=0.05,
            violinmode="overlay",
            showlegend=True,
            legend=dict(
                orientation="h", x=0.5, xanchor="center", y=1.04,
                font=dict(size=12, color="#e2e8f0"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        for ann in fig_violins.layout.annotations:
            ann.font.color = "#cbd5e1"
            ann.font.size  = 13
        fig_violins.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_violins.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")

        st.plotly_chart(fig_violins, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Descriptive stats table
        st.markdown("<div class='section-label'>Descriptive Statistics</div>", unsafe_allow_html=True)
        tab_all, tab_d, tab_nd = st.tabs(["All Patients", "Diabetic", "Non-Diabetic"])
        nice_idx = [nice[c] for c in a["num_cols"]]
        for key, t in [("all", tab_all), ("diabetic", tab_d), ("non_diabetic", tab_nd)]:
            s = a["stats"][key].rename(columns=str)
            s.index = nice_idx
            with t:
                st.dataframe(s, use_container_width=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Scatter: glucose vs HbA1c coloured by diabetes
        st.markdown("<div class='section-label'>Glucose vs HbA1c — Diabetes Separation</div>", unsafe_allow_html=True)
        _sc = a["scatter_sample"]
        fig_sc = px.scatter(
            _sc, x="HbA1c_level", y="blood_glucose_level",
            color=_sc["diabetes"].map({0: "Non-Diabetic", 1: "Diabetic"}),
            color_discrete_map={"Non-Diabetic": "#3b82f6", "Diabetic": "#ef4444"},
            opacity=0.55,
            labels={"HbA1c_level": "HbA1c Level (%)", "blood_glucose_level": "Blood Glucose (mg/dL)",
                    "color": ""},
        )
        fig_sc.update_traces(marker=dict(size=4))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=360,
                             legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08))
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────
    # TAB 2 — Categorical Analysis
    # ─────────────────────────────────────────────────
    with tab_cat:
        st.markdown("<br>", unsafe_allow_html=True)
        cat_cols = {
            "gender":          "Gender",
            "smoking_history": "Smoking History",
            "age_group":       "Age Group",
            "bmi_cat":         "BMI Category",
        }

        # Diabetes rate by each categorical feature
        st.markdown("<div class='section-label'>Diabetes Prevalence by Category</div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)

        for i, (col, label) in enumerate(cat_cols.items()):
            grp = a["cat_aggs"][col]

            fig_bar = go.Figure(go.Bar(
                x=grp["rate_pct"],
                y=grp[col],
                orientation="h",
                marker=dict(
                    color=grp["rate_pct"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#1d4ed8"], [1, "#ef4444"]],
                    showscale=False,
                ),
                text=grp["rate_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Diabetes Rate by {label}", font=dict(size=13, color="#e2e8f0")),
                xaxis=dict(title="Prevalence (%)", range=[0, grp["rate_pct"].max() * 1.25]),
                yaxis=dict(tickfont=dict(size=11)),
                height=280,
            )
            container = ca if i % 2 == 0 else cb
            with container:
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Distribution pie charts
        st.markdown("<div class='section-label'>Population Composition</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)

        for col_widget, (col, label) in zip([p1, p2, p3, p4], cat_cols.items()):
            vc = a["pie_counts"][col]
            fig_pie = go.Figure(go.Pie(
                labels=vc.index,
                values=vc.values,
                hole=0.55,
                marker=dict(colors=PALETTE[:len(vc)]),
                textinfo="percent",
                textfont=dict(size=10),
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=label, font=dict(size=12, color="#e2e8f0"), x=0.5),
                height=240,
                showlegend=True,
                legend=dict(font=dict(size=9), x=0.5, xanchor="center", y=-0.15,
                            orientation="h"),
            )
            with col_widget:
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Co-morbidities
        st.markdown("<div class='section-label'>Comorbidity Impact on Diabetes Rate</div>", unsafe_allow_html=True)
        cm1, cm2 = st.columns(2)
        for col_w, col_name, label in [
            (cm1, "hypertension", "Hypertension"),
            (cm2, "heart_disease", "Heart Disease"),
        ]:
            grp = a["comorbidity_aggs"][col_name].copy()
            grp[col_name] = grp[col_name].map({0: f"No {label}", 1: label})
            fig_cm = px.bar(grp, x=col_name, y="diabetes",
                            color=col_name, color_discrete_sequence=["#3b82f6", "#ef4444"],
                            text=grp["diabetes"].apply(lambda v: f"{v*100:.1f}%"),
                            labels={"diabetes": "Diabetes Rate", col_name: ""})
            fig_cm.update_traces(textposition="outside", textfont=dict(color="#94a3b8"))
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                                 title=dict(text=f"Diabetes Rate by {label}",
                                            font=dict(size=13, color="#e2e8f0")),
                                 yaxis=dict(tickformat=".0%"))
            with col_w:
                st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────
    # TAB 3 — Correlations
    # ─────────────────────────────────────────────────
    with tab_corr:
        st.markdown("<br>", unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown("<div class='section-label'>Pearson Correlation Matrix</div>", unsafe_allow_html=True)
        corr_cols   = a["corr_cols"]
        corr        = a["corr"]
        labels_nice = ["Age", "BMI", "HbA1c", "Glucose", "Hypertension", "Heart Disease", "Diabetes"]

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=labels_nice, y=labels_nice,
            colorscale=[[0, "#450a0a"], [0.5, "#1e293b"], [1, "#1d4ed8"]],
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
            colorbar=dict(
                thickness=14, tickfont=dict(color="#94a3b8"),
                title=dict(text="r", font=dict(color="#94a3b8")),
            ),
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=440,
                               xaxis=dict(tickfont=dict(size=11)),
                               yaxis=dict(tickfont=dict(size=11)),
                               title=dict(text="Feature Correlation Matrix",
                                          font=dict(size=14, color="#e2e8f0")))
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Diabetes correlations bar chart
        st.markdown("<div class='section-label'>Correlation with Diabetes Outcome</div>", unsafe_allow_html=True)
        d_corr = corr["diabetes"].drop("diabetes").sort_values()
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in d_corr.values]
        fig_dcorr = go.Figure(go.Bar(
            x=d_corr.values,
            y=[labels_nice[corr_cols.index(c)] for c in d_corr.index],
            orientation="h",
            marker_color=colors,
            text=d_corr.values.round(3),
            textposition="outside",
            textfont=dict(color="#94a3b8"),
        ))
        fig_dcorr.update_layout(
            **PLOTLY_LAYOUT, height=300,
            title=dict(text="Pearson r with Diabetes (1 = positive, 0 = no correlation)",
                       font=dict(size=13, color="#e2e8f0")),
            xaxis=dict(range=[-0.1, d_corr.max() * 1.3], title="Pearson r"),
        )
        st.plotly_chart(fig_dcorr, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Age vs Glucose vs Diabetes bubble
        st.markdown("<div class='section-label'>Age × Glucose × BMI (Bubble = BMI size)</div>", unsafe_allow_html=True)
        samp2 = a["bubble_sample"]
        fig_bub = px.scatter(
            samp2, x="age", y="blood_glucose_level",
            size="bmi", size_max=18,
            color=samp2["diabetes"].map({0: "Non-Diabetic", 1: "Diabetic"}),
            color_discrete_map={"Non-Diabetic": "#3b82f6", "Diabetic": "#ef4444"},
            opacity=0.5,
            labels={"age": "Age", "blood_glucose_level": "Blood Glucose (mg/dL)", "color": ""},
        )
        fig_bub.update_layout(**PLOTLY_LAYOUT, height=400,
                              legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08))
        st.plotly_chart(fig_bub, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────
    # TAB 4 — Summary & Conclusions
    # ─────────────────────────────────────────────────
    with tab_summary:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Dataset health ──
        st.markdown("<div class='section-label'>Dataset Quality</div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Missing Values</div>
                <div class="value" style="color:#22c55e">0</div>
                <div class="delta delta-pos">Clean — no imputation needed</div>
            </div>""", unsafe_allow_html=True)
        with q2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Class Imbalance</div>
                <div class="value" style="color:#f59e0b">1 : 10.3</div>
                <div class="delta delta-neg">Negative : Positive</div>
            </div>""", unsafe_allow_html=True)
        with q3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Age Range</div>
                <div class="value">0 – 80</div>
                <div class="delta">Mean: {df['age'].mean():.1f} yrs</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # ── Key findings ──
        st.markdown("<div class='section-label'>Key Findings</div>", unsafe_allow_html=True)

        insights = [
            ("HbA1c is the Strongest Predictor",
             "HbA1c level shows the highest Pearson correlation with diabetes outcomes (r ≈ 0.40). "
             "Patients with HbA1c ≥ 6.5% constitute the vast majority of diabetic cases, "
             "confirming its clinical gold-standard status for diagnosis."),
            ("Blood Glucose is a Close Second",
             "Blood glucose level carries a strong correlation (r ≈ 0.42) and provides the clearest "
             "visual separation between diabetic and non-diabetic patients. Levels above 140 mg/dL are "
             "strongly associated with positive diagnosis."),
            ("Risk Escalates Significantly with Age",
             "Diabetes prevalence rises sharply from the Young Adult to Senior age groups. "
             "Senior patients (60+) exhibit the highest prevalence rate, over 3× that of young adults, "
             "making age a critical screening criterion."),
            ("Obesity Compounds Risk Significantly",
             "Obese and Extremely Obese patients show dramatically higher diabetes rates compared to "
             "Healthy-weight individuals. The BMI category is a reliable risk stratification tool "
             "alongside clinical biomarkers."),
            ("Hypertension & Heart Disease Co-occur with Diabetes",
             "Patients with hypertension or heart disease have substantially higher diabetes prevalence, "
             "reflecting the shared metabolic syndrome pathway. These comorbidities should trigger "
             "proactive diabetes screening."),
            ("Smoking History Shows Complex Patterns",
             "Former and current smokers exhibit slightly higher diabetes rates than never-smokers, "
             "consistent with smoking's insulin resistance effects. The 'No Info' cohort mirrors "
             "the 'never' group, suggesting low reporting bias."),
        ]

        for i in range(0, len(insights), 2):
            cc1, cc2 = st.columns(2)
            with cc1:
                title, body = insights[i]
                st.markdown(f"""
                <div class="insight-card">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>""", unsafe_allow_html=True)
            if i + 1 < len(insights):
                with cc2:
                    title, body = insights[i + 1]
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{title}</h4>
                        <p>{body}</p>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # ── Recommendations ──
        st.markdown("<div class='section-label'>Clinical Recommendations</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b style="color:#93c5fd">Screening Priorities</b><br><br>
            1. <b>Prioritise HbA1c and glucose testing</b> for patients over 45, those with BMI ≥ 30, or those with hypertension/heart disease.<br>
            2. <b>Implement class-weighted modeling</b> — the 1:10 class imbalance calls for SMOTE, class weights, or threshold tuning to minimise false negatives.<br>
            3. <b>Monitor former smokers</b> as an intermediate risk cohort that is often overlooked in routine screening.<br>
            4. <b>Use age grouping as a triage filter</b> — children and teens represent a distinct low-risk cohort that may inflate model confidence in clinical tools.<br>
            5. <b>Validate model on external cohorts</b> before clinical deployment, especially across different ethnicities and healthcare settings.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI Assistant (Chatbot)
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Load project summary and diabetes document for context ──
    base = os.path.dirname(os.path.abspath(__file__))

    try:
        with open(os.path.join(base, "project_summary.md"), "r", encoding="utf-8") as _f:
            _SUMMARY = _f.read()
    except FileNotFoundError:
        _SUMMARY = """
        This is the DiabetesIQ Clinical Intelligence Platform.
        It uses an XGBoost model trained on 96,128 patient records to predict diabetes risk.
        Key features: age, BMI, HbA1c level, blood glucose, hypertension, heart disease, smoking history.
        Key findings: HbA1c ≥ 6.5% and glucose > 140 mg/dL are strongest predictors.
        The dataset has an 8.8% diabetes prevalence (class imbalance ~1:10).
        """

    try:
        with open(os.path.join(base, "diabetes_document.md"), "r", encoding="utf-8") as _f:
            _DIABETES_DOC = _f.read()
    except FileNotFoundError:
        _DIABETES_DOC = """
        Diabetes mellitus is a chronic metabolic disease characterized by elevated blood glucose.
        Types: Type 1 (autoimmune), Type 2 (insulin resistance), Gestational, Prediabetes.
        Diagnosis: HbA1c ≥ 6.5%, fasting glucose ≥ 126 mg/dL, or random glucose ≥ 200 mg/dL.
        Risk factors: obesity (BMI ≥ 30), age > 45, hypertension, heart disease, smoking, inactivity.
        """

    _SYSTEM_PROMPT = f"""You are a clinical AI assistant embedded in the DiabetesIQ platform — a diabetes risk prediction tool built with XGBoost.
Answer questions about the ML project, the dataset, diabetes risk factors, and model results.
Use the two documents below as your primary sources. Be concise, medically accurate, and helpful.
If a question falls outside the scope of the documentation, say so clearly.

--- PROJECT DOCUMENTATION ---
{_SUMMARY}
--- END PROJECT DOCUMENTATION ---

--- DIABETES CLINICAL REFERENCE ---
{_DIABETES_DOC}
--- END DIABETES CLINICAL REFERENCE ---"""

    # ── Page header ──
    st.markdown("""
    <div class="chat-header">
        <div class="page-header-inner">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
                <rect width="48" height="48" rx="14" fill="#1d4ed822"/>
                <rect x="10" y="13" width="28" height="18" rx="5" stroke="#60a5fa" stroke-width="2.2" fill="none"/>
                <circle cx="17" cy="22" r="2.2" fill="#60a5fa"/>
                <circle cx="24" cy="22" r="2.2" fill="#93c5fd"/>
                <circle cx="31" cy="22" r="2.2" fill="#60a5fa"/>
                <path d="M19 31 L15 38" stroke="#60a5fa" stroke-width="2" stroke-linecap="round"/>
                <path d="M29 31 L33 38" stroke="#60a5fa" stroke-width="2" stroke-linecap="round"/>
                <line x1="15" y1="38" x2="33" y2="38" stroke="#60a5fa" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div>
                <h1>AI Assistant</h1>
                <p>Ask anything about the diabetes prediction model, dataset, risk factors, or clinical insights.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── API key input in sidebar ──
    with st.sidebar:
        st.markdown("<div class='section-label'>OpenRouter Configuration</div>", unsafe_allow_html=True)
        _api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get a free key at openrouter.ai",
            placeholder="sk-or-v1-...",
        )
        if _api_key:
            st.markdown("<span style='color:#22c55e; font-size:0.8rem'>API key set</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#64748b; font-size:0.8rem'>Enter API key to enable chat</span>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e293b; margin: 1rem 0'>", unsafe_allow_html=True)
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # ── Initialise chat history ──
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── Suggestion chips (shown only when chat is empty) ──
    _SUGGESTIONS = [
        "What are the strongest predictors of diabetes in this dataset?",
        "How does the XGBoost model work?",
        "What does an HbA1c level of 6.5% mean clinically?",
        "How does BMI affect diabetes risk?",
        "What was the class imbalance in the training data?",
    ]

    # ── Chat message history display ──
    _chat_container = st.container()
    with _chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div class="chat-empty">
                <div class="section-title" style="color:#f1f5f9">How can I help you?</div>
                <p>Ask me about the diabetes prediction model,<br>dataset insights, or clinical risk factors.</p>
            </div>""", unsafe_allow_html=True)

            # Suggestion buttons
            st.markdown("<div class='section-label' style='text-align:center'>Try asking…</div>", unsafe_allow_html=True)
            _cols = st.columns(1)
            for _sug in _SUGGESTIONS:
                if st.button(f"{_sug}", key=f"sug_{_sug}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": _sug})
                    st.rerun()
        else:
            for _msg in st.session_state.chat_messages:
                if _msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-role-label chat-role-user">You</div>
                    <div class="chat-bubble-user">{_msg["content"]}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-role-label chat-role-assistant">DiabetesIQ Assistant</div>
                    <div class="chat-bubble-assistant">{_msg["content"]}</div>
                    """, unsafe_allow_html=True)

    # ── Chat input ──
    st.markdown("<br>", unsafe_allow_html=True)

    if not _api_key:
        st.markdown("""
        <div class="api-key-box">
            <b style="color:#93c5fd">API Key Required</b><br>
            <span style="color:#94a3b8; font-size:0.88rem">
            Enter your <a href="https://openrouter.ai" target="_blank" style="color:#3b82f6">OpenRouter</a>
            API key in the sidebar to start chatting. It's free to sign up.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        _prompt = st.chat_input("Ask about the project, model, or diabetes risk factors…")

        if _prompt:
            st.session_state.chat_messages.append({"role": "user", "content": _prompt})

        # ── Trigger AI reply whenever the last message is from the user (covers
        #    both typed input AND suggestion chip clicks) ──
        _needs_reply = (
            st.session_state.chat_messages
            and st.session_state.chat_messages[-1]["role"] == "user"
        )

        if _needs_reply:
            _client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=_api_key,
            )

            with st.spinner("Thinking…"):
                try:
                    _reply = _client.chat.completions.create(
                        model="openrouter/auto",
                        extra_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "DiabetesIQ AI Assistant",
                        },
                        messages=[{"role": "system", "content": _SYSTEM_PROMPT}]
                                 + st.session_state.chat_messages,
                        temperature=0.2,
                    ).choices[0].message.content
                except Exception as _e:
                    _reply = f"Error contacting the AI service: {_e}\n\nPlease check your API key and try again."

            st.session_state.chat_messages.append({"role": "assistant", "content": _reply})
            st.rerun()