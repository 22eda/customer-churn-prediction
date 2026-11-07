"""Configuration settings for Churn Prediction project."""
import os

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(DATA_DIR, 'customer_churn.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.joblib')
