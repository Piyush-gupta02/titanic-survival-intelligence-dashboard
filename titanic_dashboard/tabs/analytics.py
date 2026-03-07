"""Tab 6: Advanced Analytics"""

import streamlit as st
import pandas as pd
import json

from visualizations import create_correlation_heatmap
from components import insight_box


def generate_insights(df):
    """Generate automatic insights from dataset"""

    insights = []

    if df.empty:
        return insights

    f_survival = df[df["Sex"] == "female"]["Survived"].mean()
    m_survival = df[df["Sex"] == "male"]["Survived"].mean()

    if m_survival > 0 and f_survival > m_survival * 1.5:
        insights.append(
            f"🚨 **Gender Gap**: Women had {f_survival/m_survival:.1f}x higher survival "
            f"({f_survival:.0%} vs {m_survival:.0%})"
        )

    c1 = df[df["Pclass"] == 1]["Survived"].mean() if len(df[df["Pclass"] == 1]) > 0 else 0
    c3 = df[df["Pclass"] == 3]["Survived"].mean() if len(df[df["Pclass"] == 3]) > 0 else 0

    if c3 > 0 and c1 > c3 * 1.5:
        insights.append(
            f"💰 **Class Privilege**: 1st class had {c1/c3:.1f}x better survival than 3rd class"
        )

    children = df[df["Age"] <= 12]["Survived"].mean() if len(df[df["Age"] <= 12]) > 0 else 0
    adults = df[(df["Age"] > 18) & (df["Age"] < 60)]["Survived"].mean()

    if adults > 0 and children > adults:
        insights.append(
            f"👶 **Children First**: Children had {children/adults:.1f}x better survival"
        )

    return insights


def render(filtered_df, ml_data):

    st.header("📊 Advanced Analytics & Reporting")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    total = len(filtered_df)

    # ---------------------------
    # REPORT + DATA QUALITY
    # ---------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Custom Report Generator")

        st.multiselect(
            "Select Sections",
            [
                "Executive Summary",
                "Demographics",
                "ML Performance",
                "Predictions",
                "Raw Data",
            ],
            default=["Executive Summary", "Demographics"],
        )

        if st.button("📄 Generate Report"):
            st.info("PDF generation requires additional setup.")

        # ---------------------------
        # EXPORT STATISTICS
        # ---------------------------

        st.subheader("Statistics Export")

        stats = {
            "total_passengers": total,
            "survival_rate": float(filtered_df["Survived"].mean()),
            "by_class": filtered_df.groupby("Pclass")["Survived"].mean().to_dict(),
            "by_gender": filtered_df.groupby("Sex")["Survived"].mean().to_dict(),
            "avg_age": float(filtered_df["Age"].mean()),
            "avg_fare": float(filtered_df["Fare"].mean()),
            "best_model": ml_data["best_model"],
            "best_accuracy": float(ml_data["best_score"]),
        }

        st.download_button(
            "⬇️ Download Statistics (JSON)",
            json.dumps(stats, indent=2),
            "titanic_stats.json",
            "application/json",
        )

    # ---------------------------
    # DATA QUALITY REPORT
    # ---------------------------

    with col2:

        st.subheader("Data Quality Report")

        missing_age = filtered_df["Age"].isna().sum()
        missing_cabin = (filtered_df["Has_Cabin"] == 0).sum()
        duplicates = len(filtered_df) - len(filtered_df.drop_duplicates())
        fare_outliers = (filtered_df["Fare"] > 300).sum()
        age_outliers = (filtered_df["Age"] > 70).sum()

        quality = pd.DataFrame(
            {
                "Metric": [
                    "Total Records",
                    "Missing Ages",
                    "Missing Cabins",
                    "Duplicates",
                    "Outlier Fares (>300)",
                    "Outlier Ages (>70)",
                ],
                "Count": [
                    total,
                    missing_age,
                    missing_cabin,
                    duplicates,
                    fare_outliers,
                    age_outliers,
                ],
                "Percentage": [
                    "100%",
                    f"{missing_age/total*100:.1f}%",
                    f"{missing_cabin/total*100:.1f}%",
                    f"{duplicates/total*100:.1f}%",
                    f"{fare_outliers/total*100:.1f}%",
                    f"{age_outliers/total*100:.1f}%",
                ],
            }
        )

        st.dataframe(quality, hide_index=True, use_container_width=True)

    # ---------------------------
    # CORRELATION MATRIX
    # ---------------------------

    st.subheader("Feature Correlation Matrix")

    corr_cols = [
        c
        for c in ["Survived", "Pclass", "Age", "Fare", "FamilySize", "IsAlone"]
        if c in filtered_df.columns
    ]

    fig_corr = create_correlation_heatmap(filtered_df, corr_cols)

    st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------------------
    # AUTO INSIGHTS
    # ---------------------------

    st.subheader("🤖 Auto-Generated Insights")

    insights = generate_insights(filtered_df)

    if insights:
        for insight in insights:
            insight_box("Insight", insight)
    else:
        st.info("Adjust filters to see specific insights.")

    # ---------------------------
    # DATASET PREVIEW
    # ---------------------------

    st.subheader("Dataset Preview")

    preview_cols = [
        c
        for c in [
            "PassengerId",
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
            "FamilySize",
        ]
        if c in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[preview_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )