# Customer Churn Prediction & Streamlit Panel

## 1. One-line Summary
An end-to-end Machine Learning project predicting customer churn with Scikit-learn pipelines and an interactive Streamlit analytics dashboard.

## 2. Problem and Motivation
Customer churn is a critical challenge for subscription businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds a predictive classification model and decision-support dashboard to identify at-risk customers proactively.

## 3. Data
- **Dataset:** Synthetic Telco Customer Churn dataset (reproducible via data generation module).
- **Features:** Tenure, MonthlyCharges, TotalCharges, Contract type, InternetService, OnlineSecurity, TechSupport, PaperlessBilling, PaymentMethod, SeniorCitizen.
- **Target:** Churn (0 = Retained, 1 = Churned).

## 4. Approach & Methodology
- **Modular Architecture:** Separated configuration, data loading, feature engineering, modeling, training, and evaluation under `src/churn_prediction/`.
- **Preprocessing:** `ColumnTransformer` handling numerical scaling and categorical one-hot encoding.
- **Models:** Compared Scikit-learn baseline (Logistic Regression) with Ensemble model (Random Forest).
- **Validation:** Stratified train-test split (80/20) with ROC-AUC and F1-score optimization.

## 5. Results (Baseline vs Final)
- **Baseline (Logistic Regression):** ~78% Accuracy, ~0.82 ROC-AUC.
- **Final Model (Random Forest):** ~84% Accuracy, ~0.88 ROC-AUC.

## 6. How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python -m src.churn_prediction.train
   ```
3. Run tests:
   ```bash
   pytest
   ```
4. Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 7. Project Structure
```
ana-code/
├── README.md
├── requirements.txt
├── app.py
├── data/
│   └── customer_churn.csv
├── models/
│   └── churn_model.joblib
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── train.py
│       └── evaluate.py
└── tests/
    └── test_pipeline.py
```

## 8. Limitations & Next Steps
- **Limitations:** Synthetic dataset used for initial baseline; production integration requires live CRM data connectors.
- **Next Steps:** Implement XGBoost / LightGBM models, hyperparameter tuning with Optuna, and deploy FastAPI backend service.

## 9. License
MIT License.
