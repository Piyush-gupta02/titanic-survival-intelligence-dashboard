"""Tab 1: Executive Overview"""

import streamlit as st
import numpy as np

from visualizations import create_sunburst, create_gauge
from components import insight_box
from ml_engine import predict_with_confidence


def render(filtered_df, ml_data, encoders):

    st.header("📊 Executive Dashboard")

    # Safety check
    if filtered_df.empty:
        st.warning("No passengers match current filters.")
        return

    col1, col2 = st.columns([2, 1])

    # ---------------------------------------
    # SUNBURST
    # ---------------------------------------

    with col1:

        fig = create_sunburst(filtered_df)

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # RIGHT PANEL
    # ---------------------------------------

    with col2:

        insight_box(
            "🎯 Navigation Guide",
            "Click segments to drill down:<br>"
            "• Center = All passengers<br>"
            "• Ring 1 = Class<br>"
            "• Ring 2 = Gender<br>"
            "• Ring 3 = Survival<br>"
            "• Outer = Age Group"
        )

        st.subheader("📈 Current Filter Prediction")

        # ---------------------------------------
        # BUILD REPRESENTATIVE PASSENGER
        # ---------------------------------------

        try:

            filter_features = {

                "Pclass": int(filtered_df["Pclass"].mode()[0]),

                "Sex": filtered_df["Sex"].mode()[0],

                "Age": float(filtered_df["Age"].mean()),

                "SibSp": int(filtered_df["SibSp"].mode()[0]),

                "Parch": int(filtered_df["Parch"].mode()[0]),

                "Fare": float(filtered_df["Fare"].mean()),

                "Embarked": filtered_df["Embarked"].mode()[0],

                "Has_Cabin": bool(filtered_df["Has_Cabin"].mode()[0]),

                "FamilySize": int(filtered_df["FamilySize"].mode()[0]),

                "IsAlone": bool(filtered_df["IsAlone"].mode()[0]),

                "Title": "Mr",

                "AgeGroup": str(filtered_df["AgeGroup"].mode()[0]),

                "Deck": "C",

                "TicketGroupSize": 1,
            }

            result = predict_with_confidence(
                filter_features,
                ml_data,
                encoders
            )

            fig_gauge = create_gauge(result["ensemble_probability"])

            st.plotly_chart(fig_gauge, use_container_width=True)

            st.metric(
                "Actual Survival Rate",
                f"{filtered_df['Survived'].mean():.1%}"
            )

            st.metric(
                "Model Confidence",
                result["confidence_level"]
            )

        except Exception as e:

            st.info("Prediction unavailable for current filters.")