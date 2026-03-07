"""Advanced Machine Learning implementation with SHAP explainability"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb
import streamlit as st

from config import ML_CONFIG, MODEL_FEATURES


@st.cache_resource
def train_ml_pipeline(df: pd.DataFrame) -> dict:
    """Train full ML pipeline with multiple models and SHAP explainability"""

    from data_loader import encode_features, prepare_ml_features

    np.random.seed(42)

    df_encoded, encoders = encode_features(df)

    X = prepare_ml_features(df_encoded, MODEL_FEATURES)
    y = df_encoded["Survived"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    cv_scores = {}

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    with st.spinner("🌲 Training Random Forest..."):

        rf = RandomForestClassifier(**ML_CONFIG["random_forest"])

        rf.fit(X_train, y_train)

        rf_calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=5)

        rf_calibrated.fit(X_train, y_train)

        # SHAP explainer
        rf_explainer = shap.TreeExplainer(rf)

        models["random_forest"] = {
            "model": rf_calibrated,
            "raw_model": rf,
            "explainer": rf_explainer,
            "score": rf_calibrated.score(X_test, y_test),
            "probabilities": rf_calibrated.predict_proba(X_test)[:, 1],
        }

        cv_scores["random_forest"] = cross_val_score(
            rf,
            X_train,
            y_train,
            cv=5
        ).mean()

    # =====================================================
    # XGBOOST
    # =====================================================

    with st.spinner("⚡ Training XGBoost..."):

        xgb_model = xgb.XGBClassifier(**ML_CONFIG["xgboost"])

        xgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        models["xgboost"] = {
            "model": xgb_model,
            "score": xgb_model.score(X_test_scaled, y_test),
            "probabilities": xgb_model.predict_proba(X_test_scaled)[:, 1],
            "feature_importance": xgb_model.feature_importances_,
        }

        cv_scores["xgboost"] = cross_val_score(
            xgb_model,
            X_train_scaled,
            y_train,
            cv=3
        ).mean()

    # =====================================================
    # GRADIENT BOOSTING
    # =====================================================

    with st.spinner("🚀 Training Gradient Boosting..."):

        gb = GradientBoostingClassifier(**ML_CONFIG["gradient_boosting"])

        gb.fit(X_train, y_train)

        models["gradient_boosting"] = {
            "model": gb,
            "score": gb.score(X_test, y_test),
            "probabilities": gb.predict_proba(X_test)[:, 1],
        }

        cv_scores["gradient_boosting"] = cross_val_score(
            gb,
            X_train,
            y_train,
            cv=5
        ).mean()

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    with st.spinner("📈 Training Logistic Regression..."):

        lr = LogisticRegression(**ML_CONFIG["logistic_regression"])

        lr.fit(X_train_scaled, y_train)

        models["logistic_regression"] = {
            "model": lr,
            "score": lr.score(X_test_scaled, y_test),
            "probabilities": lr.predict_proba(X_test_scaled)[:, 1],
            "coefficients": lr.coef_[0],
        }

        cv_scores["logistic_regression"] = cross_val_score(
            lr,
            X_train_scaled,
            y_train,
            cv=5
        ).mean()

    # =====================================================
    # SVM
    # =====================================================

    with st.spinner("🎯 Training SVM..."):

        svm = SVC(**ML_CONFIG["svm"])

        svm.fit(X_train_scaled, y_train)

        models["svm"] = {
            "model": svm,
            "score": svm.score(X_test_scaled, y_test),
            "probabilities": svm.predict_proba(X_test_scaled)[:, 1],
        }

        cv_scores["svm"] = cross_val_score(
            svm,
            X_train_scaled,
            y_train,
            cv=5
        ).mean()

    # =====================================================
    # SOFT VOTING ENSEMBLE
    # =====================================================

    with st.spinner("🎪 Building Soft Voting Ensemble..."):

        voting = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(**ML_CONFIG["random_forest"])),
                ("xgb", xgb.XGBClassifier(**ML_CONFIG["xgboost"])),
                ("gb", GradientBoostingClassifier(**ML_CONFIG["gradient_boosting"])),
                ("lr", LogisticRegression(**ML_CONFIG["logistic_regression"]))
            ],
            voting="soft"
        )

        voting.fit(X_train_scaled, y_train)

        models["soft_ensemble"] = {
            "model": voting,
            "score": voting.score(X_test_scaled, y_test),
            "probabilities": voting.predict_proba(X_test_scaled)[:, 1],
        }

    # =====================================================
    # STACKING ENSEMBLE
    # =====================================================

    with st.spinner("🏗️ Building Stacking Ensemble..."):

        stacking = StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(**ML_CONFIG["random_forest"])),
                ("xgb", xgb.XGBClassifier(**ML_CONFIG["xgboost"])),
                ("gb", GradientBoostingClassifier(**ML_CONFIG["gradient_boosting"]))
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )

        stacking.fit(X_train_scaled, y_train)

        models["stacking_ensemble"] = {
            "model": stacking,
            "score": stacking.score(X_test_scaled, y_test),
            "probabilities": stacking.predict_proba(X_test_scaled)[:, 1],
        }

    # =====================================================
    # ROC + PR CURVES
    # =====================================================

    roc_data = {}
    pr_data = {}

    for name, model_info in models.items():

        if "probabilities" not in model_info:
            continue

        probs = model_info["probabilities"]

        fpr, tpr, _ = roc_curve(y_test, probs)

        roc_data[name] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr)
            }

        precision, recall, _ = precision_recall_curve(y_test, probs)

        pr_data[name] = {
                "precision": precision,
                "recall": recall,
                "avg_precision": average_precision_score(y_test, probs)
            }
    roc_data= dict(sorted(roc_data.items(), key=lambda x: x[1]["auc"], reverse=True))
    pr_data = dict(sorted(pr_data.items(), key=lambda x: x[1]["avg_precision"], reverse=True))
    
    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    importance_df = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "random_forest": models["random_forest"]["raw_model"].feature_importances_,
        "xgboost": models["xgboost"]["feature_importance"],
        "gradient_boosting": models["gradient_boosting"]["model"].feature_importances_
    })

    importance_df["average"] = importance_df[
        ["random_forest", "xgboost", "gradient_boosting"]
    ].mean(axis=1)

    importance_df = importance_df.sort_values("average", ascending=False)

    best_model_name = max(models, key=lambda x: models[x]["score"])

    return {
        "models": models,
        "cv_scores": cv_scores,
        "feature_importance": importance_df,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "best_model": best_model_name,
        "best_score": models[best_model_name]["score"],
        "shap_explainer": rf_explainer
    }


# =========================================================
# PREDICTION WITH CONFIDENCE + SHAP
# =========================================================

def predict_with_confidence(features_dict: dict, ml_data: dict, encoders: dict) -> dict:
    """Predict survival probability with confidence and SHAP explanation"""

    feature_vector = np.array([[
        features_dict["Pclass"],
        encoders["Sex"].transform([features_dict["Sex"]])[0],
        features_dict["Age"],
        features_dict["SibSp"],
        features_dict["Parch"],
        features_dict["Fare"],
        encoders["Embarked"].transform([features_dict["Embarked"]])[0],
        int(features_dict["Has_Cabin"]),
        features_dict["FamilySize"],
        int(features_dict["IsAlone"]),
        encoders["Title"].transform([features_dict["Title"]])[0],
        encoders["AgeGroup"].transform([features_dict["AgeGroup"]])[0],
        features_dict["Fare"] / features_dict["FamilySize"],
        int(features_dict["Age"] <= 12),
        0,
        encoders["Deck"].transform([features_dict.get("Deck", "Unknown")])[0],
        features_dict.get("TicketGroupSize", 1)
    ]])

    feature_scaled = ml_data["scaler"].transform(feature_vector)

    all_probs = {}

    for name, model_info in ml_data["models"].items():

        model = model_info["model"]

        if name in ["logistic_regression", "svm", "soft_ensemble", "stacking_ensemble", "xgboost"]:
            prob = model.predict_proba(feature_scaled)[0][1]
        else:
            prob = model.predict_proba(feature_vector)[0][1]

        all_probs[name] = prob

    weights = {
        "random_forest": 0.25,
        "xgboost": 0.30,
        "gradient_boosting": 0.20,
        "logistic_regression": 0.15,
        "svm": 0.10
    }

    weighted_prob = sum(
        all_probs[name] * weights[name]
        for name in weights if name in all_probs
    )

    probs_array = np.array(list(all_probs.values()))

    confidence_interval = (
        float(np.percentile(probs_array, 2.5)),
        float(np.percentile(probs_array, 97.5))
    )

    std_prob = np.std(probs_array)

    if std_prob < 0.05:
        confidence_level = "Very High"
    elif std_prob < 0.10:
        confidence_level = "High"
    elif std_prob < 0.15:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    risk_level = (
        "Low" if weighted_prob > 0.7
        else "Medium" if weighted_prob > 0.4
        else "High"
    )

    # SHAP Explanation
    shap_values = None

    explainer = ml_data.get("shap_explainer")

    if explainer is not None:
        shap_values = explainer.shap_values(feature_vector)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = shap_values[0]

    return {
        "ensemble_probability": weighted_prob,
        "all_predictions": all_probs,
        "confidence_interval": confidence_interval,
        "confidence_level": confidence_level,
        "risk_level": risk_level,
        "model_disagreement": std_prob,
        "shap_values": shap_values
    }