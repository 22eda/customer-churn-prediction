"""Unit tests for data pipeline and models."""
import pytest
import pandas as pd
from churn_prediction.data import generate_synthetic_churn_data, get_train_test_data, load_data
from churn_prediction.models import get_baseline_model, get_random_forest_model

def test_data_generation():
    df = generate_synthetic_churn_data(n_samples=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert 'Churn' in df.columns

def test_train_test_split():
    total_len = len(load_data())
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.2)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == total_len

def test_baseline_model():
    model = get_baseline_model()
    assert model is not None

def test_random_forest_model():
    model = get_random_forest_model()
    assert model is not None
