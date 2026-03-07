"""Configuration and constants for Titanic Dashboard"""

import os

# ===============================
# Paths
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = r"E:\New Eda\data\Titanic-Dataset.csv"

# ===============================
# Dashboard settings
# ===============================

DASHBOARD_CONFIG = {
    "page_title": "Titanic Survival Intelligence Pro",
    "page_icon": "🚢",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# ===============================
# ML Configuration
# ===============================

ML_CONFIG = {
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_samples_split': 5,
        'subsample': 0.8,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 250,
        'max_depth': 6,
        'learning_rate': 0.03,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'logistic_regression': {
        'C': 1.0,
        'class_weight': 'balanced',
        'max_iter': 2000,
        'random_state': 42,
        'solver': 'lbfgs'
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42
    }
}

AVAILABLE_MODELS = [
    "random_forest",
    "logistic_regression",
    "xgboost",
    "gradient_boosting",
    "svm"
]

# ===============================
# Feature Engineering
# ===============================

MODEL_FEATURES = [
    'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked_encoded', 'Has_Cabin', 'FamilySize', 'IsAlone',
    'Title_encoded', 'AgeGroup_encoded', 'FarePerPerson',
    'IsChild', 'IsMother', 'Deck_encoded', 'TicketGroupSize'
]

VIZ_FEATURES = [
    'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize'
]

# ===============================
# Colors
# ===============================

SURVIVAL_PALETTE = {0: '#e74c3c', 1: '#2ecc71'}

CLASS_COLORS = ['#3498db', '#9b59b6', '#e67e22']

MODEL_COLORS = {
    "random_forest": "#FFFFFF",        # white (very visible)
    "xgboost": "#FF4B4B",              # red
    "gradient_boosting": "#FFC107",    # amber
    "logistic_regression": "#00E5FF",  # cyan
    "svm": "#AB47BC",                  # purple
    "soft_ensemble": "#4CAF50",        # green
    "stacking_ensemble": "#FF7043"     # orange
}
# ===============================
# Visualization order
# ===============================

DECK_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Unknown']

# ===============================
# Tabs
# ===============================

TABS = [
    "📊 Overview",
    "🤖 ML Insights",
    "🧮 3D Visualizer",
    "⏱️ Timeline",
    "🔮 Predictor",
    "🧠 Explainable AI",
    "📈 Analytics",
    "🕸️ Network"
]