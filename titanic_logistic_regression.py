# ------------------------------------------------------------
# Project 4: Logistic Regression Model Implementation and Evaluation
# Binary Classification: Titanic Survival Prediction
# ------------------------------------------------------------

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# 2. Load Dataset
df = pd.read_csv("titanic.csv")

print("Initial dataset shape:", df.shape)
print(df.head())

# 3. Data Preprocessing

# Drop columns that are not useful for prediction or have too many missing values
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Define features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Logistic Regression Model Training
log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
log_reg.fit(X_train, y_train)

# 6. Predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# 7. Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did Not Survive", "Survived"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Titanic Logistic Regression")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 9. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Logistic Regression").plot()
plt.title("ROC Curve - Titanic Logistic Regression")
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# 10. Interpretation
print("\n--- Interpretation ---")
print(f"Accuracy {accuracy:.2%} shows the percentage of correct predictions.")
print(f"Precision {precision:.2%} indicates how often predicted survivors actually survived.")
print(f"Recall {recall:.2%} shows how well the model identifies actual survivors.")
print(f"F1-score {f1:.2%} balances precision and recall.")
print(f"AUC {auc:.2%} reflects the model's ability to separate survivors from non-survivors.")
