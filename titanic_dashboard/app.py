"""
Titanic Survival BI Dashboard - Main Entry Point
Professional multi-file architecture with advanced ML
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time
import pandas as pd
import io

# Module imports
from titanic_dashboard.config import DATA_PATH, TABS
from titanic_dashboard.data_loader import load_and_engineer_data, encode_features
from titanic_dashboard.ml_engine import train_ml_pipeline
from titanic_dashboard.styles import apply_dark_theme

# Tab imports
from titanic_dashboard.tabs import overview, ml_insights, visualizer, timeline, predictor, explainable_ai, analytics, network


# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="Titanic Survival Intelligence Pro",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_dark_theme()

import joblib

@st.cache_resource
def load_models():
    return joblib.load("ml_models.pkl")

# -----------------------------------------
# DATA LOADING
# -----------------------------------------

try:

    with st.spinner("🚢 Loading Titanic dataset and ML models..."):

        time.sleep(0.5)

        df = load_and_engineer_data(DATA_PATH)
        df_encoded, encoders = encode_features(df)

        ml_data = load_models()

except FileNotFoundError:

    st.error(f"Dataset not found at path: {DATA_PATH}")
    st.stop()
# -----------------------------------------
# SIDEBAR
# -----------------------------------------

st.sidebar.markdown("""
<div style="text-align:center; padding:20px 0;">
<h1 style="margin:0;">🚢 Titanic BI Pro</h1>
<p style="color:#bdc3c7;">Advanced ML Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.header("🔍 Global Filters")

selected_class = st.sidebar.multiselect(
    "Class",
    [1, 2, 3],
    default=[1, 2, 3],
    format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'}"
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    ["male", "female"],
    default=["male", "female"]
)

selected_port = st.sidebar.multiselect(
    "Port",
    ["S", "C", "Q"],
    default=["S", "C", "Q"],
    format_func=lambda x: {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }[x]
)

age_range = st.sidebar.slider(
    "Age",
    0,
    80,
    (0, 80)
)

fare_range = st.sidebar.slider(
    "Fare ($)",
    0.0,
    500.0,
    (0.0, 500.0)
)

# -----------------------------------------
# APPLY FILTERS
# -----------------------------------------

filtered_df = df[
    (df["Pclass"].isin(selected_class)) &
    (df["Sex"].isin(selected_gender)) &
    (df["Embarked"].isin(selected_port)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Fare"].between(fare_range[0], fare_range[1]))
]

st.sidebar.info(f"**Filtered:** {len(filtered_df):,} passengers")

if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()


# -----------------------------------------
# EXPORT OPTIONS
# -----------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("💾 Export Data")

export_format = st.sidebar.selectbox(
    "Format",
    ["CSV", "Excel", "JSON"]
)

if export_format == "CSV":

    csv = filtered_df.to_csv(index=False)

    st.sidebar.download_button(
        "⬇️ Download CSV",
        csv,
        "titanic_filtered.csv",
        "text/csv"
    )

elif export_format == "Excel":

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False)

    st.sidebar.download_button(
        "⬇️ Download Excel",
        buffer.getvalue(),
        "titanic_filtered.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:

    json_str = filtered_df.to_json(orient="records", indent=2)

    st.sidebar.download_button(
        "⬇️ Download JSON",
        json_str,
        "titanic_filtered.json",
        "application/json"
    )


# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown("""
<div style="text-align:center; padding:20px 0;">
<h1 style="font-size:3em;">🚢 Titanic Survival Intelligence Pro</h1>
<p style="font-size:1.2em; color:#bdc3c7;">
Advanced ML • 6 Models • Confidence Intervals • 3D Analytics
</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------
# KPI METRICS
# -----------------------------------------

cols = st.columns(5)

survival_rate = filtered_df["Survived"].mean() if len(filtered_df) > 0 else 0
female_ratio = (filtered_df["Sex"] == "female").mean() if len(filtered_df) > 0 else 0
avg_fare = filtered_df["Fare"].mean() if len(filtered_df) > 0 else 0

metrics = [
    ("👥 Total", f"{len(filtered_df):,}"),
    ("❤️ Survival", f"{filtered_df['Survived'].mean():.1%}"),
    ("🎯 Best ML Score", f"{ml_data['best_score']:.1%}", ml_data['best_model']),
    ("👩 Female", f"{(filtered_df['Sex']=='female').mean():.1%}"),
    ("💰 Avg Fare", f"${filtered_df['Fare'].mean():.0f}")
]

for col, (label, value, *help_text) in zip(cols, metrics):

    with col:

        st.metric(
            label,
            value,
            help=help_text[0] if help_text else None
        )


st.markdown("---")


# -----------------------------------------
# TABS
# -----------------------------------------

tabs = st.tabs(TABS)

with tabs[0]:
    overview.render(filtered_df, ml_data, encoders)

with tabs[1]:
    ml_insights.render(ml_data)

with tabs[2]:
    visualizer.render(filtered_df)

with tabs[3]:
    timeline.render(filtered_df)

with tabs[4]:
    predictor.render(filtered_df, ml_data, encoders)

with tabs[5]:
    explainable_ai.render(filtered_df, ml_data)

with tabs[6]:
    analytics.render(filtered_df, ml_data)

with tabs[7]:
    network.render(filtered_df)


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#9aa0a6; font-size:14px; padding:10px;">
🚢 <b>Titanic Survival Intelligence Pro</b> |
Models: RF + XGB + GB + LR + SVM + Ensembles |
Built with Streamlit + Plotly + Scikit-Learn
</div>
""", unsafe_allow_html=True)