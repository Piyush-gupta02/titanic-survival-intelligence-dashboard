import joblib
from titanic_dashboard.data_loader import load_and_engineer_data
from titanic_dashboard.ml_engine import train_ml_pipeline
from titanic_dashboard.config import DATA_PATH

df = load_and_engineer_data(DATA_PATH)

ml_data = train_ml_pipeline(df)

joblib.dump(ml_data, "ml_models.pkl")

print("Models saved")