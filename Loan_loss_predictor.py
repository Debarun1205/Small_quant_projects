# Probability of Default (PD) Model + Expected Loss Function
# ---------------------------------------------------------
# Assumption:
# Recovery Rate = 10%
# Loss Given Default (LGD) = 90%
# Expected Loss = PD * LGD * Exposure at Default
# Exposure at Default = loan_amt_outstanding

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Target column
target = "default"

# Drop ID column (not predictive in real-world modeling)
X = df.drop(columns=[target, "customer_id"])
y = df[target]

# ---------------------------------------------------------
# 2. Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)

# ---------------------------------------------------------
# 4. Model (Logistic Regression)
# ---------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate Model
# ---------------------------------------------------------
pd_probs = model.predict_proba(X_test)[:, 1]
preds = (pd_probs >= 0.5).astype(int)

print("AUC Score:", roc_auc_score(y_test, pd_probs))
print("Brier Score:", brier_score_loss(y_test, pd_probs))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ---------------------------------------------------------
# 6. Function to Predict PD
# ---------------------------------------------------------
def predict_pd(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Returns Probability of Default (PD)
    """
    
    data = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score
    }])

    pd_estimate = model.predict_proba(data)[0, 1]
    return pd_estimate


# ---------------------------------------------------------
# 7. Function to Calculate Expected Loss
# ---------------------------------------------------------
def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=0.10
):
    """
    Expected Loss = PD × LGD × EAD
    LGD = 1 - recovery_rate
    EAD = loan_amt_outstanding
    """
    
    pd_est = predict_pd(
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    )

    lgd = 1 - recovery_rate
    ead = loan_amt_outstanding

    loss = pd_est * lgd * ead
    return loss


# ---------------------------------------------------------
# 8. Example Usage
# ---------------------------------------------------------
sample_pd = predict_pd(
    credit_lines_outstanding=3,
    loan_amt_outstanding=5000,
    total_debt_outstanding=12000,
    income=55000,
    years_employed=5,
    fico_score=620
)

sample_loss = expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=5000,
    total_debt_outstanding=12000,
    income=55000,
    years_employed=5,
    fico_score=620
)

print("Predicted PD:", round(sample_pd, 4))
print("Expected Loss:", round(sample_loss, 2))
