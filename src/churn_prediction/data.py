"""Data loading and synthetic dataset generation for Churn Prediction."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, DATA_PATH, DATA_DIR

def generate_synthetic_churn_data(n_samples: int = 2000, save_path: str = DATA_PATH) -> pd.DataFrame:
    """Generates a statistically robust synthetic customer churn dataset."""
    np.random.seed(RANDOM_STATE)
    
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, size=n_samples)
    
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.35, 0.50, 0.15])
    online_security = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.4, 0.6])
    tech_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.35, 0.65])
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples, p=[0.35, 0.2, 0.25, 0.2])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    
    churn_prob = (
        0.3 * (contract_type == 'Month-to-month') +
        0.2 * (internet_service == 'Fiber optic') +
        0.15 * (online_security == 'No') +
        0.15 * (tech_support == 'No') -
        0.005 * tenure +
        0.002 * monthly_charges - 0.2
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob * 4))
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_type,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'SeniorCitizen': senior_citizen,
        'Churn': churn
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads customer churn data, generating synthetic data if file does not exist."""
    if not os.path.exists(path):
        return generate_synthetic_churn_data(n_samples=2500, save_path=path)
    return pd.read_csv(path)

def get_train_test_data(path: str = DATA_PATH, test_size: float = 0.2):
    """Splits data into stratified train and test sets."""
    df = load_data(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
