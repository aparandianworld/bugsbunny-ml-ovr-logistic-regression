import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()
df = pd.DataFrame(data = digits.data, columns = [f"pixel_{i}" for i in range(digits.data.shape[1])]) # (1797, 64)
df['target'] = digits.target

# Remove missing or impute values
print("Original shape: ", df.shape)
print("Number of missing values: \n", df.isnull().sum())
df = df.dropna(how = 'any')
print("Shape after removing missing values: ", df.shape)

# Dataset statistics
print("\nDataset statistics: ")
print(df.describe())

# Preview data
print("\nPreview data: ")
print(df.head())

# Feature matrix and target vector
feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])] # 64 pixels 
target_name = "target"
X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("\nFeature matrix: ", X.shape)
print("Target vector: ", y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-vs-Rest (OVR) Logistic Regression
# FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7
# ovr_model = LogisticRegression(multi_class = 'ovr', max_iter = 1000)
ovr_model = OneVsRestClassifier(LogisticRegression(max_iter = 1000))
ovr_model.fit(X_train, y_train)

# Make predictions
y_train_hat = ovr_model.predict(X_train)
y_test_hat = ovr_model.predict(X_test)

# Evaluate
print("\nEvaluation:")
print("\n-----------------------------------")
print("Accuracy (OvR): ", accuracy_score(y_test, y_test_hat))
print("Confusion matrix (OvR): \n", confusion_matrix(y_test, y_test_hat))
print("Classification report (OvR): \n", classification_report(y_test, y_test_hat))