"""Feature engineering and preprocessing pipeline construction."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline() -> ColumnTransformer:
    """Constructs a ColumnTransformer for numerical and categorical features."""
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )
    return preprocessor
