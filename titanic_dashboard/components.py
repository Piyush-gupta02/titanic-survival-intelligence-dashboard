"""Reusable UI components for Titanic Dashboard"""

import streamlit as st
import pandas as pd


# ---------------------------------------------------
# INSIGHT BOX
# ---------------------------------------------------

def insight_box(title: str, content: str, box_type: str = "success"):
    """Render colored insight box"""

    colors = {
        "success": ("rgba(46, 204, 113, 0.15)", "#2ecc71"),
        "warning": ("rgba(231, 76, 60, 0.15)", "#e74c3c"),
        "info": ("rgba(52, 152, 219, 0.15)", "#3498db"),
        "ml": ("rgba(155, 89, 182, 0.15)", "#9b59b6"),
    }

    bg, border = colors.get(box_type, colors["info"])

    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:4px solid {border};
            padding:15px;
            border-radius:0 10px 10px 0;
            margin:10px 0;
        ">
        <h4 style="margin:0;color:white;">{title}</h4>
        <p style="margin:5px 0 0 0;color:#e0e0e0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------
# PASSENGER CARD
# ---------------------------------------------------

def passenger_card(passenger: pd.Series, col):
    """Display passenger card in column"""

    survived = passenger.get("Survived", 0) == 1
    sex = passenger.get("Sex", "Unknown")
    age = passenger.get("Age", 0)
    pclass = passenger.get("Pclass", "-")
    fare = passenger.get("Fare", 0)

    with col:
        st.markdown(
            f"""
            <div style="
                background:{'rgba(46,204,113,0.2)' if survived else 'rgba(231,76,60,0.2)'};
                border-radius:10px;
                padding:15px;
                text-align:center;
                border:2px solid {'#2ecc71' if survived else '#e74c3c'};
            ">

            <h4 style="margin:0;color:white;">{str(sex).title()}</h4>

            <p style="margin:5px 0;color:#bdc3c7;">
                {age:.0f} yrs | Class {pclass}
            </p>

            <p style="margin:5px 0;color:#bdc3c7;">
                ${fare:.0f}
            </p>

            <h3 style="margin:10px 0;color:{'#2ecc71' if survived else '#e74c3c'}">
                {"SURVIVED" if survived else "DIED"}
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------
# RISK BADGES
# ---------------------------------------------------

def risk_badge(factors: list):
    """Display risk factors nicely"""

    if factors:

        st.markdown("**⚠ Risk Factors:**")

        for factor in factors:
            st.markdown(
                f"<span style='color:#e74c3c'>{factor}</span>",
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            "<span style='color:#2ecc71'>✅ Low risk profile</span>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------
# MODEL PERFORMANCE TABLE
# ---------------------------------------------------

def model_performance_table(models: dict, cv_scores: dict) -> pd.DataFrame:
    """Create model comparison dataframe"""

    best_model = max(models, key=lambda x: models[x]["score"])

    data = []

    for name, info in models.items():

        data.append(
            {
                "Model": name.replace("_", " ").title(),
                "Test Accuracy": f"{info['score']:.3f}",
                "CV Score": (
                    f"{cv_scores[name]:.3f}"
                    if name in cv_scores
                    else "N/A"
                ),
                "Status": "🏆 Best Model" if name == best_model else ""
            }
        )

    df = pd.DataFrame(data)

    return df.sort_values("Test Accuracy", ascending=False)