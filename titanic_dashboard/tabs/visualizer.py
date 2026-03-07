"""Tab 3: 3D Visualizer"""

import streamlit as st
import pandas as pd

from visualizations import create_3d_scatter, create_pca_visualization
from components import insight_box


def render(filtered_df):
    """Render 3D visualization tab"""

    st.header("🧮 3D Interactive Visualizations")

    st.markdown(
        """
Explore passenger relationships using **interactive 3D visualizations**.
Rotate, zoom, and inspect how variables like **Age, Fare, Class, and Gender**
influenced survival patterns.
"""
    )

    viz_type = st.selectbox(
        "Select Visualization",
        [
            "Age-Fare-Class Scatter",
            "Class-Gender-Age Bubble",
            "PCA Dimensionality Reduction"
        ]
    )

    # ---------------------------------------------------
    # AGE FARE CLASS SCATTER
    # ---------------------------------------------------

    if viz_type == "Age-Fare-Class Scatter":

        fig = create_3d_scatter(
            filtered_df,
            x="Age",
            y="Fare",
            z="Pclass"
        )

        fig.update_layout(height=700)

        st.plotly_chart(fig, use_container_width=True)

        insight_box(
            "🎮 Interactive Controls",
            "• <b>Rotate</b>: Click and drag<br>"
            "• <b>Zoom</b>: Scroll wheel<br>"
            "• <b>Pan</b>: Right-click drag<br>"
            "• <b>Hover</b>: Inspect passenger attributes<br>"
            "<b>Green</b> = Survived, <b>Red</b> = Did Not Survive"
        )

    # ---------------------------------------------------
    # CLASS GENDER AGE BUBBLE
    # ---------------------------------------------------

    elif viz_type == "Class-Gender-Age Bubble":

        plot_df = filtered_df.copy()

        plot_df["Sex_num"] = (plot_df["Sex"] == "female").astype(int)

        fig = create_3d_scatter(
            plot_df,
            x="Pclass",
            y="Sex_num",
            z="Age",
            size="Fare"
        )

        fig.update_layout(
            height=700,
            scene=dict(
                yaxis=dict(
                    ticktext=["Male", "Female"],
                    tickvals=[0, 1],
                    title="Gender"
                )
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------

    else:

        fig = create_pca_visualization(filtered_df)

        fig.update_layout(height=700)

        st.plotly_chart(fig, use_container_width=True)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        features = filtered_df[
            ["Pclass", "Age", "Fare", "FamilySize"]
        ].copy()

        features["Sex"] = (filtered_df["Sex"] == "female").astype(int)

        features = features.fillna(features.mean())

        scaler = StandardScaler()

        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=3)

        pca.fit(features_scaled)

        explained_variance = pca.explained_variance_ratio_.sum()

        st.markdown(
            f"**Total variance explained by 3 components: {explained_variance:.2%}**"
        )

        weights_df = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2", "PC3"],
            index=["Pclass", "Age", "Fare", "FamilySize", "Sex"]
        )

        st.write("PCA Component Weights:")

        st.dataframe(weights_df.round(3), use_container_width=True)