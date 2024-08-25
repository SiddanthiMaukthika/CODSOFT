# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE  # Ensure imbalanced-learn is installed

# Load the dataset
data = pd.read_csv('C:/Users/sidda/Downloads/archive (2)/creditcard.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Preprocess the data
features = data.drop('Class', axis=1)
target = data['Class']

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Handle class imbalance with SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(scaled_features, target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Build and train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Build and train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict and evaluate Logistic Regression model
y_pred_logistic = logistic_model.predict(X_test)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

print("Logistic Regression Model:")
print(f"Precision: {precision_logistic}")
print(f"Recall: {recall_logistic}")
print(f"F1 Score: {f1_logistic}")
print("Confusion Matrix:")
print(conf_matrix_logistic)

# Predict and evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Model:")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_rf}")
print("Confusion Matrix:")
print(conf_matrix_rf)

# ROC Curve and AUC for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# ROC Curve and AUC for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_logistic, tpr_logistic, color='darkorange', lw=2, label=f'Logistic Regression (area = {roc_auc_logistic:.2f})')
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the model
joblib.dump(logistic_model, 'logistic_fraud_detection_model.pkl')
joblib.dump(rf_model, 'rf_fraud_detection_model.pkl')

# Load and use the model
loaded_logistic_model = joblib.load('logistic_fraud_detection_model.pkl')
loaded_rf_model = joblib.load('rf_fraud_detection_model.pkl')

# Example of making predictions with new data
# new_data = pd.DataFrame({'Feature1': [value1], 'Feature2': [value2], ...})
# new_data_scaled = scaler.transform(new_data)
# predictions_logistic = loaded_logistic_model.predict(new_data_scaled)
# predictions_rf = loaded_rf_model.predict(new_data_scaled)

