"""Model training and serialization script."""
import os
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from .data import get_train_test_data
from .models import get_random_forest_model
from .config import MODEL_PATH, MODEL_DIR

def train_and_save_model():
    """Trains the churn prediction model and saves it to disk."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    print("Training Random Forest model...")
    model = get_random_forest_model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train_and_save_model()
