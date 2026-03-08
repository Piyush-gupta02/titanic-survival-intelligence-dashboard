"""Tab 4: Evacuation Timeline"""

import streamlit as st
import pandas as pd
import plotly.express as px

from visualizations import create_animated_timeline, create_deck_survival_map
from components import insight_box


def render(filtered_df):

    st.header("⏱️ Evacuation Timeline Animation")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    col1, col2 = st.columns([1, 3])

    # ------------------------------------
    # SETTINGS PANEL
    # ------------------------------------

    with col1:

        st.subheader("Animation Settings")

        speed = st.slider(
            "Speed (ms)",
            100,
            2000,
            500
        )

        show_decks = st.multiselect(
            "Show Decks",
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            default=['A', 'B', 'C', 'D', 'E']
        )

        insight_box(
            "🎬 Timeline Story",
            "<b>0-60 min</b>: 1st class evacuation<br>"
            "<b>60-90 min</b>: 2nd class & women/children<br>"
            "<b>90-120 min</b>: General panic<br>"
            "<b>120+ min</b>: Final lifeboats"
        )

    # ------------------------------------
    # ANIMATION
    # ------------------------------------

    with col2:

        timeline_df = filtered_df[
            filtered_df["Deck"].isin(show_decks)
        ].copy()

        if timeline_df.empty:

            st.info("No passengers on selected decks.")

        else:

            fig = create_animated_timeline(
                timeline_df,
                speed
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ------------------------------------
    # SURVIVAL BY TIME
    # ------------------------------------

    st.subheader("Survival Rate by Evacuation Timing")

    bins = [0, 30, 60, 90, 120, 150, 180]

    labels = [
        "0-30min",
        "30-60min",
        "60-90min",
        "90-120min",
        "120-150min",
        "150-180min"
    ]

    time_bins = pd.cut(
        filtered_df["Evacuation_Time"],
        bins=bins,
        labels=labels
    )

    time_survival = (
        filtered_df
        .groupby(time_bins, observed=True)["Survived"]
        .mean()
        .reset_index()
    )

    fig_time = px.bar(
        time_survival,
        x="Evacuation_Time",
        y="Survived",
        color="Survived",
        color_continuous_scale="RdYlGn",
        text=time_survival["Survived"].apply(lambda x: f"{x:.1%}"),
        title="Survival Rate by Time Window"
    )

    fig_time.update_traces(
        textposition="outside"
    )

    fig_time.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        yaxis=dict(tickformat=".0%")
    )

    st.plotly_chart(
        fig_time,
        use_container_width=True
    )

    # ------------------------------------
    # DECK SURVIVAL
    # ------------------------------------

    st.subheader("🚢 Ship Deck Survival Map")

    fig_deck = create_deck_survival_map(
        filtered_df
    )

    st.plotly_chart(
        fig_deck,
        use_container_width=True
    )