"""Tab 5: Survival Predictor"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from ml_engine import predict_with_confidence
from visualizations import create_scenario_comparison
from components import passenger_card, risk_badge
from visualizations import create_scenario_comparison

def get_age_group(age):
    """Match AgeGroup used during training"""

    if age <= 5:
        return "Infant"
    elif age <= 12:
        return "Child"
    elif age <= 18:
        return "Teen"
    elif age <= 25:
        return "Young Adult"
    elif age <= 35:
        return "Adult"
    elif age <= 50:
        return "Middle Age"
    elif age <= 65:
        return "Senior"
    else:
        return "Elder"


def render(filtered_df, ml_data, encoders):

    st.header("🔮 Advanced Survival Predictor with Confidence")

    col1, col2, col3 = st.columns(3)

    # ---------------------------
    # PASSENGER PROFILE
    # ---------------------------

    with col1:

        st.subheader("Passenger Profile")

        pred_class = st.selectbox(
            "Class",
            [1, 2, 3],
            format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class"
        )

        pred_gender = st.selectbox("Gender", ["female", "male"])

        pred_age = st.slider("Age", 0, 80, 25)

        pred_title = st.selectbox(
            "Title",
            ["Mr", "Mrs", "Miss", "Master", "Rare"]
        )

    # ---------------------------
    # TRAVEL DETAILS
    # ---------------------------

    with col2:

        st.subheader("Travel Details")

        pred_fare = st.slider("Fare ($)", 0, 600, 50)

        pred_sibsp = st.slider("Siblings/Spouses", 0, 8, 0)

        pred_parch = st.slider("Parents/Children", 0, 6, 0)

        pred_port = st.selectbox(
            "Port",
            ["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown"
            }[x]
        )

    # ---------------------------
    # ACCOMMODATIONS
    # ---------------------------

    with col3:

        st.subheader("Accommodations")

        pred_cabin = st.checkbox("Has Cabin Record", value=False)

        pred_deck = st.selectbox(
            "Deck",
            ["A", "B", "C", "D", "E", "F", "G", "Unknown"],
            index=7
        )

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------

    family_size = pred_sibsp + pred_parch + 1

    is_alone = 1 if family_size == 1 else 0

    features = {
        "Pclass": pred_class,
        "Sex": pred_gender,
        "Age": pred_age,
        "SibSp": pred_sibsp,
        "Parch": pred_parch,
        "Fare": pred_fare,
        "Embarked": pred_port,
        "Has_Cabin": pred_cabin,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": pred_title,
        "AgeGroup": get_age_group(pred_age),
        "Deck": pred_deck,
        "TicketGroupSize": 1,
    }

    # ---------------------------
    # PREDICTION
    # ---------------------------

    result = predict_with_confidence(
        features,
        ml_data,
        encoders
    )

    probs = result["all_predictions"]

    # ---------------------------
    # DISPLAY RESULT
    # ---------------------------

    color = (
        "#2ecc71"
        if result["ensemble_probability"] > 0.6
        else "#f1c40f"
        if result["ensemble_probability"] > 0.4
        else "#e74c3c"
    )

    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        border: 3px solid {color};
        text-align: center;">

        <h3>Ensemble Prediction</h3>

        <h1 style="color:{color}">
        {result['ensemble_probability']:.1%}
        </h1>

        <p>Confidence: <b>{result['confidence_level']}</b></p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # INDIVIDUAL MODEL OUTPUT
    # ---------------------------

    with st.expander("See Individual Model Predictions"):

        for model, prob in probs.items():

            st.progress(
                float(prob),
                text=f"{model.replace('_',' ').title()}: {prob:.1%}"
            )

    # ---------------------------
    # MODEL AGREEMENT
    # ---------------------------

    st.subheader("Model Agreement Analysis")

    prob_values = list(probs.values())

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=prob_values,
            name="Model Predictions",
            marker_color="#3498db"
        )
    )

    fig.add_hline(
        y=result["ensemble_probability"],
        line_dash="dash",
        annotation_text="Ensemble",
        line_color="#2ecc71"
    )

    fig.update_layout(
        yaxis=dict(tickformat=".0%"),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # RISK FACTORS
    # ---------------------------

    risk_factors = []

    if pred_gender == "male":
        risk_factors.append("⚠️ Male")

    if pred_class == 3:
        risk_factors.append("⚠️ 3rd Class")

    if pred_age > 50:
        risk_factors.append("⚠️ Senior Age")

    if family_size > 4:
        risk_factors.append("⚠️ Large Family")

    if pred_fare < 20:
        risk_factors.append("⚠️ Low Fare")

    risk_badge(risk_factors)

    # ---------------------------
    # SIMILAR PASSENGERS
    # ---------------------------
    st.subheader("Scenario Comparison")

    scenarios = [
    ("Current Passenger", result["ensemble_probability"], None),
    ("Female 1st Class", 0.82, None),
    ("Male 3rd Class", 0.18, None),
    ("Child 2nd Class", 0.65, None)]

    fig = create_scenario_comparison(scenarios)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Similar Historical Passengers")

    df_copy = filtered_df.copy()

    similar_features = df_copy[
        ["Pclass", "Sex", "Age", "Fare", "FamilySize"]
    ].copy()

    similar_features["Sex"] = (
        similar_features["Sex"] == "female"
    ).astype(int)

    input_vector = np.array([
        pred_class,
        1 if pred_gender == "female" else 0,
        pred_age,
        pred_fare,
        family_size
    ])

    distances = np.sqrt(
        ((similar_features.values - input_vector) ** 2).sum(axis=1)
    )

    df_copy["Distance"] = distances

    similar_passengers = df_copy.nsmallest(5, "Distance")

    cols = st.columns(5)

    for idx, (_, passenger) in enumerate(similar_passengers.iterrows()):

        passenger_card(passenger, cols[idx])