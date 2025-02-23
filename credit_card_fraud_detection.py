import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load the dataset
try:
    data = pd.read_csv('creditcard.csv', delimiter='\t')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'creditcard.csv' was not found.")
    exit()

# Display dataset information
print(data.info())
print(data.head())

# Ensure 'Class' column exists
if 'Class' not in data.columns:
    print("Error: 'Class' column not found in dataset.")
    exit()

# Check for missing values
print(data.isnull().sum())

# 1 Class Distribution - Check Data Imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=data, palette="coolwarm")
plt.title("Class Distribution: Fraud vs Non-Fraud")
plt.xticks([0, 1], ["Genuine (0)", "Fraud (1)"])
plt.show()

# 2 Box Plot for Transaction Amount (Fraud vs Genuine)
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=data, palette="coolwarm")
plt.title("Transaction Amount Distribution for Fraud & Genuine")
plt.xticks([0, 1], ["Genuine (0)", "Fraud (1)"])
plt.show()

# 3 Time vs Amount Scatter Plot - Identify unusual transactions
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data['Time'], y=data['Amount'], hue=data['Class'], alpha=0.5, palette={0: "blue", 1: "red"})
plt.xlabel("Time (seconds)")
plt.ylabel("Transaction Amount")
plt.title("Time vs Transaction Amount (Fraud vs Genuine)")
plt.legend(["Genuine", "Fraud"])
plt.show()

# 4 Distribution of Transaction Time
plt.figure(figsize=(10, 4))
sns.histplot(data[data['Class'] == 0]['Time'], bins=50, kde=True, color='blue', label="Genuine")
sns.histplot(data[data['Class'] == 1]['Time'], bins=50, kde=True, color='red', label="Fraud")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Transaction Time for Fraud & Genuine")
plt.legend()
plt.show()

# 5 Correlation Heatmap - Identify feature relationships
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Separate features and target
X = data.drop(columns=['Class'], errors='ignore')  # Drop safely
y = data['Class']

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize 'Amount' and 'Time' features (if present)
scaler = StandardScaler()
if 'Amount' in X_resampled.columns and 'Time' in X_resampled.columns:
    X_resampled[['Amount', 'Time']] = scaler.fit_transform(X_resampled[['Amount', 'Time']])

# 6 Pair Plot of Key Features
sampled_df = X_resampled.sample(500, random_state=42)  # Sample for better visualization
sns.pairplot(sampled_df[['V1', 'V2', 'V3', 'V4', 'V5']], diag_kind="kde")
plt.suptitle("Pair Plot of Important Features", y=1.02)
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Train Random Forest with hyperparameter tuning
rf_clf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_clf = grid_search.best_estimator_

# Evaluate Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))

# 7 Confusion Matrix Heatmap for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluate Random Forest
y_pred_rf = best_rf_clf.predict(X_test)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 8 Confusion Matrix Heatmap for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap="Greens")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9 Feature Importance for Random Forest
feature_importances = pd.Series(best_rf_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
feature_importances[:10].plot(kind='bar', color='purple')
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

# Compute ROC curves
log_fpr, log_tpr, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, best_rf_clf.predict_proba(X_test)[:, 1])

# 10 ROC Curve Comparison
plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]):.2f})', linestyle='--')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, best_rf_clf.predict_proba(X_test)[:, 1]):.2f})', linestyle='-')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
