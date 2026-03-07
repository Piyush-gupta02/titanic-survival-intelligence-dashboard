"""Tab: Explainable AI (SHAP)"""

import streamlit as st
import shap
import matplotlib.pyplot as plt
plt.style.use("dark_background")

def render(filtered_df, ml_data):

    st.header("🧠 Explainable AI with SHAP")

    st.markdown(
        """
        SHAP (SHapley Additive Explanations) explains **how each feature contributes
        to a prediction**. Positive values increase survival probability, negative
        values decrease it.
        """
    )

    # Load trained model + explainer
    explainer = ml_data["shap_explainer"]
    X_test = ml_data["X_test"]

    # ------------------------------
    # GLOBAL IMPORTANCE
    # ------------------------------

    st.subheader("Global Feature Importance")

    from config import MODEL_FEATURES

    shap_values = explainer(X_test)
    shap_values.feature_names = MODEL_FEATURES
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    fig_summary = plt.figure()
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(fig_summary)

    st.divider()

    # ------------------------------
    # LOCAL EXPLANATION
    # ------------------------------

    st.subheader("Local Prediction Explanation (Waterfall)")

    passenger_index = st.slider(
        "Select Passenger Example",
        0,
        len(X_test) - 1,
        0
    )

    fig_waterfall = plt.figure()

    shap.plots.waterfall(
        shap_values[passenger_index],
        max_display=10,
        show=False
    )

    st.pyplot(fig_waterfall)