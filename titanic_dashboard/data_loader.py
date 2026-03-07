"""Advanced data loading and feature engineering for Titanic Dashboard"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st


@st.cache_data(ttl=3600)
def load_and_engineer_data(filepath: str) -> pd.DataFrame:
    """
    Load Titanic dataset and perform advanced feature engineering.
    
    Args:
        filepath (str): Path to dataset
        
    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataframe
    """

    np.random.seed(42)

    # -----------------------------
    # Load raw dataset
    # -----------------------------
    df = pd.read_csv(filepath)

    # =============================
    # MISSING VALUE IMPUTATION
    # =============================

    # Age: median grouped by Pclass & Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Embarked: mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare: median grouped by Pclass
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # =============================
    # FEATURE ENGINEERING
    # =============================

    # Cabin presence
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    # Family structure
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Economic indicator
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # -----------------------------
    # Title extraction
    # -----------------------------

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].fillna('Unknown')

    title_mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }

    df['Title'] = df['Title'].replace(title_mapping)

    # -----------------------------
    # Age Groups
    # -----------------------------

    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
        labels=[
            'Infant', 'Child', 'Teen',
            'Young Adult', 'Adult',
            'Middle Age', 'Senior', 'Elder'
        ]
    )

    # -----------------------------
    # Child / Mother Indicators
    # -----------------------------

    df['IsChild'] = (df['Age'] <= 12).astype(int)

    df['IsMother'] = (
        (df['Sex'] == 'female') &
        (df['Parch'] > 0) &
        (df['Age'] > 18) &
        (df['Title'].isin(['Mrs', 'Rare']))
    ).astype(int)

    # -----------------------------
    # Deck Extraction
    # -----------------------------

    df['Deck'] = df['Cabin'].astype(str).str[0]
    df['Deck'] = df['Deck'].replace('n', 'Unknown')

    # -----------------------------
    # Ticket Group Size
    # -----------------------------

    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

    # =============================
    # TIMELINE SIMULATION
    # =============================

    n = len(df)

    base_time = np.random.normal(90, 30, n)

    time_adjustments = (
        (df['Pclass'] == 1) * -30 +
        (df['Pclass'] == 3) * 20 +
        (df['Sex'] == 'female') * -15 +
        (df['IsChild'] == 1) * -20 +
        (df['IsAlone'] == 1) * 10 +
        np.random.normal(0, 10, n)
    )

    df['Evacuation_Time'] = base_time + time_adjustments
    df['Evacuation_Time'] = df['Evacuation_Time'].clip(10, 180)

    df['TimeBin'] = (df['Evacuation_Time'] // 15) * 15

    return df


def encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode categorical variables for ML models.
    
    Returns:
        df_encoded, encoders
    """

    df_encoded = df.copy()
    encoders = {}

    categorical_cols = [
        ('Sex', 'Sex_encoded'),
        ('Embarked', 'Embarked_encoded'),
        ('Title', 'Title_encoded'),
        ('AgeGroup', 'AgeGroup_encoded'),
        ('Deck', 'Deck_encoded')
    ]

    for col, new_col in categorical_cols:
        le = LabelEncoder()
        df_encoded[new_col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


def prepare_ml_features(df: pd.DataFrame, feature_list: list) -> np.ndarray:
    """
    Prepare feature matrix for ML models.
    
    Args:
        df: dataframe
        feature_list: list of ML feature columns
        
    Returns:
        numpy feature matrix
    """

    X = df[feature_list].copy()

    # Handle any remaining missing values
    X = X.fillna(X.median(numeric_only=True))

    return X.values