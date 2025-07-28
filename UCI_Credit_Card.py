# Project: Predicting Credit Card Default Using Logistic Regression

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc)

# Load the Dataset
df = pd.read_csv('UCI_Credit_Card.csv')
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())
print(df.describe())

# Preprocessing and Cleaning
# Drop the unnecessary columns
df.drop('ID',axis=1,inplace=True)
# Check for any Null Values and find out how many null values exist in the dataset
print(df.isnull().sum())
# Inspect Data Types
print(df.dtypes)
# Check Target Distribution
# This helps us understand the class balance, whether or not we will need to address imbalances
print(df['default.payment.next.month'].value_counts())
sns.countplot(x='default.payment.next.month',data=df)
plt.title("Class Distribution (Default: 1, No Default: 0)")
plt.show()
# Visualize Distributions for Key Features
# This helps us to spot suspicious spikes, long tails  or zero inflated features
numeric_features = ['LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
for col in numeric_features:
    plt.figure(figsize=(10,10))
    sns.histplot(df[col],bins=100, label=col, kde=True)
    plt.title("Distribution of " + col)
    plt.legend()
    plt.show()
# Box-plots to Spot Outliers
for col in numeric_features:
    plt.figure(figsize=(10,10))
    sns.boxplot(x=df[col])
    plt.title("Boxplot of " + col)
    plt.show()
# Statistical Outlier Detection
outlier_summary = {}
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] > upper) | (df[col] < lower)).sum()
    outlier_summary[col] = n_outliers
    outlier_summary[col + '_std'] = df[col].std()
    outlier_summary[col + '_mean'] = df[col].mean()
    outlier_summary[col + '_median'] = df[col].median()
    outlier_summary[col + '_min'] = df[col].min()
    outlier_summary[col + '_max'] = df[col].max()
    print(f"{col}: {n_outliers} outliers detected (lower: {lower:.1f}, upper: {upper:.1f})")
# Check for impossible or Suspicious values
for col in numeric_features:
    n_negative = (df[col] < 0 ).sum()
    if n_negative > 0:
        print(f"{col}: {n_negative} negative outliers detected")
# Class Imbalance Check Again
sns.countplot(x='default.payment.next.month', data=df)
plt.title("Class Distribution (Default: 1, No Default: 0)")
plt.show()
# Correction Heatmap(Feature Redundancy Check)
plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

# Feature Engineering
# Check unique values
print(df['SEX'].unique())
print(df['EDUCATION'].unique())
print(df['MARRIAGE'].unique())
# Clean rare categories
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
# One-Hot Encoding
# Why? Logistic regression can technically handle numeric-coded categories,
# but one-hot encoding is sometimes better, especially if categories are not ordinal.
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
# Create New Features
df['UTILIZATION_RATIO'] = df['BILL_AMT1'] / df['LIMIT_BAL']
df['AVG_PAY_AMT'] = df[[f'PAY_AMT{i}' for i in range(1, 7)]].mean(axis=1)
df['AVG_BILL_AMT'] = df[[f'BILL_AMT{i}' for i in range(1, 7)]].mean(axis=1)
# Check for nulls
print(df.isnull().sum())

# Test-Train Split
# Separate features and target
X = df.drop('default.payment.next.month', axis = 1 )
y = df ['default.payment.next.month']
# Stratify=y ensure that the class balance is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target Distribution:\n ", y_train.value_counts(normalize=True))
print("Test target Distribution:\n ", y_test.value_counts(normalize=True))

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
# Training the Model
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
# Tell Logistic Regression to pay more attention to the minority class:
# model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
# Prediction and Printing the Evaluation Results
y_predict = model.predict(X_test)
y_predict_proba = model.predict_proba(X_test)[:,1] # Probabilities for ROC
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))
# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_predict_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr,tpr,label=f'ROC curve(area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credict Default Prediction')
plt.legend(loc='lower right')
plt.show()


# Train Logistic Regression Model
# Training the Model
# Tell Logistic Regression to pay more attention to the minority class:
model02 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model02.fit(X_train, y_train)
# Evaluation
# Prediction and Printing the Evaluation Results
y_predict = model02.predict(X_test)
y_predict_proba = model02.predict_proba(X_test)[:,1] # Probabilities for ROC
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))
# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_predict_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr,tpr,label=f'ROC curve(area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credict Default Prediction for Logistic Regression Model to pay more attention to the minority class')
plt.legend(loc='lower right')
plt.show()
