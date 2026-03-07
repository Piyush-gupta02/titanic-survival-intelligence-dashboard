"""Tab 2: ML Insights"""

import streamlit as st

from visualizations import (
    create_ml_comparison,
    create_feature_importance,
    create_confusion_matrix
)

from components import insight_box, model_performance_table


def render(ml_data):

    st.header("🤖 Advanced ML Analysis")

    # ---------------------------------------------------
    # MODEL PERFORMANCE TABLE
    # ---------------------------------------------------

    st.subheader("Model Performance Summary")

    perf_df = model_performance_table(
        ml_data["models"],
        ml_data["cv_scores"]
    )

    st.dataframe(
        perf_df,
        hide_index=True,
        use_container_width=True
    )

    # ---------------------------------------------------
    # ROC & PR CURVES
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("ROC Curves")

        fig_roc = create_ml_comparison(
            ml_data["roc_data"],
            "roc"
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:

        st.subheader("Precision-Recall Curves")

        fig_pr = create_ml_comparison(
            ml_data["pr_data"],
            "pr"
        )

        st.plotly_chart(fig_pr, use_container_width=True)

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------

    st.subheader("Feature Importance")

    fig_imp = create_feature_importance(
        ml_data["feature_importance"],
        top_n=12
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------------------------------------------
    # INSIGHT BOX
    # ---------------------------------------------------

    insight_box(
        "🧠 ML Insights",
        "<b>Sex</b> is the strongest predictor (~30% importance)<br>"
        "<b>Pclass</b> and <b>Fare</b> together explain a large portion of survival variance<br>"
        "<b>Ensemble models</b> reduce variance and improve reliability",
        "ml"
    )

    # ---------------------------------------------------
    # CONFUSION MATRICES
    # ---------------------------------------------------

    st.subheader("Confusion Matrices")

    cols = st.columns(3)

    top_models = ["random_forest", "xgboost", "soft_ensemble"]

    for col, model_name in zip(cols, top_models):

        if model_name not in ml_data["models"]:
            continue

        with col:

            model = ml_data["models"][model_name]["model"]

            try:

                # Some models require scaled features
                if model_name in [
                    "logistic_regression",
                    "svm",
                    "soft_ensemble",
                    "stacking_ensemble",
                    "xgboost"
                ]:

                    X_input = ml_data["scaler"].transform(
                        ml_data["X_test"]
                    )

                else:

                    X_input = ml_data["X_test"]

                y_pred = model.predict(X_input)

                fig_cm = create_confusion_matrix(
                    ml_data["y_test"],
                    y_pred,
                    model_name.replace("_", " ").title()
                )

                st.plotly_chart(
                    fig_cm,
                    use_container_width=True
                )

            except Exception:

                st.info(f"{model_name} predictions unavailable.")