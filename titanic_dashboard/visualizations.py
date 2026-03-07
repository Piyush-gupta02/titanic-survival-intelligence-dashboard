"""Centralized visualization functions"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from config import SURVIVAL_PALETTE, CLASS_COLORS, MODEL_COLORS, DECK_ORDER


# ---------------------------------------------------
# SUNBURST
# ---------------------------------------------------

def create_sunburst(df: pd.DataFrame) -> go.Figure:
    """Hierarchical survival analysis"""

    fig = px.sunburst(
        df,
        path=["Pclass", "Sex", "Survived", "AgeGroup"],
        values="PassengerId",
        color="Survived",
        color_discrete_map=SURVIVAL_PALETTE,
        title="Hierarchical Survival Analysis"
    )

    fig.update_layout(
        height=700,
        paper_bgcolor="black",
        font_color="white"
    )
    fig.update_traces(
    insidetextorientation="radial")
    return fig


# ---------------------------------------------------
# 3D SCATTER
# ---------------------------------------------------

def create_3d_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color: str = "Survived",
    size: str = "FamilySize"
) -> go.Figure:
    """Generic reusable 3D scatter plot"""

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color,
        size=size if size in df.columns else None,
        hover_data=[c for c in ["Sex", "Embarked", "Title"] if c in df.columns],
        color_discrete_map=SURVIVAL_PALETTE,
        opacity=0.7,
        title=f"3D: {x} × {y} × {z}"
    )

    fig.update_layout(
        scene=dict(
            bgcolor="black",
            xaxis=dict(
                showbackground=True,
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            )
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        height=700
    )

    return fig

# ---------------------------------------------------
# ML COMPARISON (ROC / PR)
# ---------------------------------------------------

def create_ml_comparison(data_dict: dict, metric: str = "roc") -> go.Figure:
    """Compare ML models using ROC or Precision-Recall curves"""

    fig = go.Figure()

    for name, data in data_dict.items():

        color = MODEL_COLORS.get(name, "#ffffff")

        if metric == "roc":

            fig.add_trace(
                go.Scatter(
                    x=data["fpr"],
                    y=data["tpr"],
                    name=f"{name.replace('_',' ').title()} (AUC={data['auc']:.3f})",
                    line=dict(color=color, width=4)
                )
            )

        else:

            fig.add_trace(
                go.Scatter(
                    x=data["recall"],
                    y=data["precision"],
                    name=f"{name.replace('_',' ').title()} (AP={data['avg_precision']:.3f})",
                    line=dict(color=color, width=4)
                )
            )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    if metric == "roc":

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Random Classifier",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash")
            )
        )

        title = "ROC Curve Comparison"
        x_title = "False Positive Rate"
        y_title = "True Positive Rate"

    else:

        title = "Precision-Recall Curve Comparison"
        x_title = "Recall"
        y_title = "Precision"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),

        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.15)",
    zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False
    )

    return fig


# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

def create_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10
) -> go.Figure:
    """Feature importance chart"""

    df = importance_df.head(top_n)

    fig = px.bar(
        df,
        x="average",
        y="feature",
        orientation="h",
        color="average",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Feature Importance"
    )

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        yaxis_title="",
        xaxis_title="Average Importance"
    )

    return fig


# ---------------------------------------------------
# GAUGE
# ---------------------------------------------------

def create_gauge(probability: float, title: str = "Survival Probability") -> go.Figure:
    """Probability gauge"""

    color = (
        "#2ecc71" if probability > 0.6
        else "#f1c40f" if probability > 0.4
        else "#e74c3c"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": title, "font": {"color": "white", "size": 16}},
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white", "tickvals": [0, 25, 50, 75, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.1)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 33], "color": "rgba(231,76,60,0.2)"},
                    {"range": [33, 66], "color": "rgba(241,196,15,0.2)"},
                    {"range": [66, 100], "color": "rgba(46,204,113,0.2)"}
                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    return fig


# ---------------------------------------------------
# PCA
# ---------------------------------------------------

def create_pca_visualization(df: pd.DataFrame) -> go.Figure:
    """3D PCA projection"""

    features = df[["Pclass", "Age", "Fare", "FamilySize"]].copy()

    features["Sex"] = (df["Sex"] == "female").astype(int)

    features = features.fillna(features.mean())

    pca = PCA(n_components=3)

    pca_result = pca.fit_transform(features)

    pca_df = pd.DataFrame(
        pca_result,
        columns=["PC1", "PC2", "PC3"]
    )

    pca_df["Survived"] = df["Survived"].values

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Survived",
        color_discrete_map=SURVIVAL_PALETTE,
        title=f"PCA Projection ({pca.explained_variance_ratio_.sum():.1%} variance)",
        labels={
        "PC1": "Socioeconomic Status (Fare + Class)",
        "PC2": "Age & Family Structure",
        "PC3": "Gender Influence"
    }
)

    fig.update_layout(
        scene=dict(bgcolor="black", xaxis=dict(
            showbackground=True,
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),yaxis=dict(
            showbackground=True,
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ), zaxis=dict(
            showbackground=True,
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        )),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        height=700
    )

    return fig


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> go.Figure:

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Died", "Survived"],
        y=["Died", "Survived"],
        color_continuous_scale="RdBu_r",
        title=f"{model_name} Confusion Matrix"
    )

    fig.update_layout(
        paper_bgcolor="black",
        font_color="white"
    )

    return fig


# ---------------------------------------------------
# SCENARIO COMPARISON
# ---------------------------------------------------

def create_scenario_comparison(scenarios: list) -> go.Figure:
    """Compare prediction scenarios"""

    fig = go.Figure()

    colors = list(MODEL_COLORS.values())

    for i, (name, prob, details) in enumerate(scenarios):

        fig.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[prob],
                marker_color=colors[i % len(colors)],
                text=f"{prob:.1%}",
                textposition="auto"
            )
        )

    fig.update_layout(
        title="Scenario Comparison",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        paper_bgcolor="black",
        font_color="white",
        showlegend=False
    )

    return fig


# ---------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------

def create_correlation_heatmap(
    df: pd.DataFrame,
    cols: list
) -> go.Figure:

    corr = df[cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )

    fig.update_layout(
        paper_bgcolor="black",
        font_color="white"
    )

    return fig

def create_animated_timeline(df: pd.DataFrame, speed: int = 500) -> go.Figure:
    """Animated evacuation timeline"""
    df = df.sort_values("TimeBin")
    fig = px.scatter(
        df,
        x='Evacuation_Time',
        y='Deck',
        color='Survived',
        size='Fare',
        hover_data=['Sex', 'Age', 'Pclass'],
        animation_frame='TimeBin',
        animation_group='PassengerId',
        color_discrete_map=SURVIVAL_PALETTE,
        title="Evacuation Timeline",
        labels={'Evacuation_Time': 'Minutes After Impact'}
    )

    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        height=500,
        xaxis=dict(range=[0, 180]),
        yaxis=dict(categoryorder='array', categoryarray=DECK_ORDER)
    )
    fig.update_traces(marker=dict(opacity=0.8))
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = speed
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = speed // 2

    return fig

def create_deck_survival_map(df: pd.DataFrame) -> go.Figure:
    """Deck survival visualization"""

    deck_data = df[df["Deck"] != "T"].groupby("Deck").agg({
        'Survived': 'mean',
        'PassengerId': 'count'
    }).reset_index()

    deck_data.columns = ['Deck', 'SurvivalRate', 'Count']

    colors = [
        '#2ecc71' if s > 0.5 else '#e74c3c'
        for s in deck_data['SurvivalRate']
    ]

    fig = go.Figure()

    for idx, row in deck_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Deck']],
            y=[row['SurvivalRate']],
            marker_color=colors[idx],
            text=f"{row['SurvivalRate']:.1%}<br>(n={int(row['Count'])})",
            textposition='auto'
        ))

    fig.update_layout(
        title="Survival Rate by Ship Deck",
        yaxis=dict(tickformat='.0%', title='Survival Rate'),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        showlegend=False
    )

    return fig