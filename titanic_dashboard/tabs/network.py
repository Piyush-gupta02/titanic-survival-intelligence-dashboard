"""Tab 7: Network Analysis"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render(filtered_df):

    st.header("🕸️ Family & Social Network Analysis")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # --------------------------------------
    # WORK ON COPY (DO NOT MODIFY ORIGINAL)
    # --------------------------------------

    df = filtered_df.copy()

    df["LastName"] = df["Name"].str.split(",").str[0]

    # --------------------------------------
    # FAMILY AGGREGATION
    # --------------------------------------

    families = (
        df[df["FamilySize"] > 1]
        .groupby("LastName")
        .agg(
            SurvivalRate=("Survived", "mean"),
            Size=("Survived", "count"),
            Survivors=("Survived", "sum"),
            AvgClass=("Pclass", "mean"),
            AvgFare=("Fare", "mean"),
            FamilySize=("FamilySize", "first"),
        )
        .reset_index()
        .rename(columns={"LastName": "Family"})
    )

    families = (
        families[families["Size"] >= 2]
        .sort_values("Size", ascending=False)
        .head(20)
    )

    if families.empty:
        st.info("No families found for current filters.")
        return

    col1, col2 = st.columns([2, 1])

    # --------------------------------------
    # FAMILY BUBBLE CHART
    # --------------------------------------

    with col1:

        fig = px.scatter(
            families,
            x="Size",
            y="SurvivalRate",
            size="AvgFare",
            color="SurvivalRate",
            color_continuous_scale='RdYlGn',
            hover_data=["Family", "AvgClass"],
            title="Family Survival Analysis (Top 20 Families)",
            labels={
                "Size": "Family Size",
                "SurvivalRate": "Family Survival Rate",
            },
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------
    # FAMILY INSIGHTS
    # --------------------------------------

    with col2:

        st.markdown(
            """
            <div style="background: rgba(46, 204, 113, 0.15);
            border-left: 4px solid rgba(46,204,113,0.6);
            padding: 15px;
            border-radius: 0 10px 10px 0;">

            <h4 style="margin:0;">👨‍👩‍👧‍👦 Family Insights</h4>

            <p style="margin-top:5px;">
            • Large families often had mixed survival<br>
            • Wealthier families had higher survival rates<br>
            • Some families all survived, others all perished
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------
        # SURVIVAL DISTRIBUTION
        # --------------------------------------

        fam_dist = families["SurvivalRate"].value_counts(
            bins=4
        ).sort_index()

        if not fam_dist.empty:

            fig_pie = px.pie(
                values=fam_dist.values,
                names=[
                    f"{int(i.left*100)}-{int(i.right*100)}%"
                    for i in fam_dist.index
                ],
                title="Family Survival Distribution",
            )

            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    # --------------------------------------
    # FAMILY TABLE
    # --------------------------------------

    st.subheader("Family Details")

    st.dataframe(
        families[
            [
                "Family",
                "Size",
                "Survivors",
                "SurvivalRate",
                "AvgClass",
                "AvgFare",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )