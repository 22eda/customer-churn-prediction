"""Model definitions for Scikit-learn baseline and advanced classifiers."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from .features import get_preprocessing_pipeline
from .config import RANDOM_STATE

def get_baseline_model() -> Pipeline:
    """Creates a baseline Logistic Regression pipeline."""
    preprocessor = get_preprocessing_pipeline()
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

def get_random_forest_model() -> Pipeline:
    """Creates a Random Forest classifier pipeline."""
    preprocessor = get_preprocessing_pipeline()
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
