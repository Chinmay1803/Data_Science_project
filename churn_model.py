import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# -----------------------
# Load data (robust CSV)
# -----------------------
df = pd.read_csv(
    "data.csv",
    encoding="latin1",
    sep=None,
    engine="python"
)

# Drop ID column
df.drop(columns=["customerID"], errors="ignore", inplace=True)

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------
# Feature / target split
# -----------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -----------------------
# Preprocessing pipeline
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# -----------------------
# Model
# -----------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Train
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -----------------------
# Save model
# -----------------------
joblib.dump(pipeline, "churn_model.pkl")

print("✅ Model saved as churn_model.pkl")
