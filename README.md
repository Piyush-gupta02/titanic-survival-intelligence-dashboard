# 🚢 Titanic Survival Intelligence Dashboard

- An interactive Machine Learning and Data Analytics dashboard that explores survival patterns from the famous Titanic disaster dataset.

- This project combines data analysis, predictive modeling, interactive visualizations, and explainable AI to understand the factors influencing passenger survival.

### 🔗 Live Dashboard:
https://titanic-survival-dashboard.streamlit.app

---

# 📊 Project Overview

The **RMS Titanic sank on April 15, 1912**, after colliding with an iceberg during its maiden voyage.

- Total passengers and crew: **2224**
- Survivors: **~38%**
- Fatalities: **~62%**

The disaster revealed strong survival patterns related to:

- Passenger class
- Gender
- Age
- Family size
- Ticket fare

This project transforms the Titanic dataset into an **interactive analytics dashboard** that allows users to explore survival patterns and predict survival probability using machine learning models.

---

# 📂 Dataset

Source: **Kaggle Titanic Dataset**

The dataset contains passenger information including demographics, ticket details, and survival status.

| Feature | Description |
|------|-------------|
PassengerId | Unique passenger identifier |
Survived | Survival status (0 = No, 1 = Yes) |
Pclass | Passenger class (1st, 2nd, 3rd) |
Name | Passenger name |
Sex | Gender |
Age | Passenger age |
SibSp | Number of siblings/spouses aboard |
Parch | Number of parents/children aboard |
Ticket | Ticket number |
Fare | Ticket price |
Cabin | Cabin number |
Embarked | Port of embarkation |

---

# 🤖 Machine Learning Models

The dashboard compares multiple machine learning models to predict survival probability.

Models implemented:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine
- Ensemble Model (Soft Voting)

Evaluation metrics include:

- ROC Curve
- Precision-Recall Curve
- AUC Score
- Confusion Matrix

---

# 🔍 Dashboard Features

### 📊 Executive Dashboard
Provides high-level statistics and survival insights from the dataset.

### 🤖 Machine Learning Insights
Compare ML models using ROC curves and Precision-Recall curves.

### 🌐 Interactive 3D Visualizations
Explore relationships between variables using **3D scatter plots and PCA projections**.

### ⏳ Evacuation Timeline Animation
Animated visualization showing how survival patterns evolved during evacuation.

### 🎯 Survival Predictor
Users can input passenger characteristics and receive **predicted survival probability**.

### 🧠 Explainable AI (SHAP)
Understand how different features influence predictions.

### 📜 Similar Passenger Analysis
Find historical passengers with similar characteristics and compare survival outcomes.

---

# 🏗️ Project Structure
Titanic-Eda Project
│
├── data
│   └── Titanic-Dataset.csv
│
├── notebook
│   └── Titanic_Notebook.ipynb
│
├── titanic_dashboard
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── ml_engine.py
│   ├── styles.py
│   ├── visualizations.py
│   ├── components.py
│   └── tabs
│       ├── overview.py
│       ├── predictor.py
│       ├── ml_insights.py
│       ├── timeline.py
│       └── ...
│
├── train_models.py
├── ml_models.pkl
├── requirements.txt
└── README.md

---

# 🛠 Technologies Used

- Python
- Streamlit
- Plotly
- Scikit-Learn
- XGBoost
- SHAP
- Pandas
- NumPy

---

# 🚀 Running the Project Locally

Clone the repository
git clone https://github.com/Piyush-gupta02/titanic-survival-intelligence-dashboard.git


## Install dependencies
- pip install -r requirements.txt

## Run the dashboard
- streamlit run titanic_dashboard\app.py


---

# 🌍 Live Dashboard

https://titanic-survival-dashboard.streamlit.app

---

# 📈 Key Insights

Some interesting patterns discovered from the dataset:

- Women had significantly higher survival rates than men
- First-class passengers had better survival chances
- Children were prioritized during evacuation
- Higher ticket fares correlated with higher survival probability
- Family size influenced survival outcomes

---

# 👨‍💻 Author

**Piyush Gupta**

Software Developer | Machine Learning Enthusiast

GitHub  
https://github.com/Piyush-gupta02

---

# ⭐ Support

If you like this project, consider **starring the repository** ⭐

