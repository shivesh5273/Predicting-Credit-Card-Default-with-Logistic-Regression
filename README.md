# Predicting Credit Card Default with Logistic Regression

A comprehensive machine learning project using the UCI Credit Card dataset to predict the likelihood of credit card default.  
This repository includes the full workflow: data exploration, feature engineering, model building, handling class imbalance, and professional reporting.

---

## 📁 Files Included

- `UCI_Credit_Card.py` – Complete Python code for all data preparation, EDA, modeling, and evaluation steps.
- `UCI_Credit_Card.csv` – Original dataset (if allowed; otherwise, see [Data Source](#data-source)).
- `UCI_Credit_Card_Report.pdf` / `.docx` – Full technical report, including results and interpretations.
- `figures/roc_curve_standard.png` – ROC Curve (Standard Model)
- `figures/roc_curve_balanced.png` – ROC Curve (Class-balanced Model)
- `figures/corr_heatmap.png` – Correlation Heatmap

---

## 📊 Project Overview

- **Goal:** Predict whether a credit card holder will default on their next payment.
- **Techniques Used:**  
  - Exploratory Data Analysis (EDA)  
  - Outlier detection and feature engineering  
  - Logistic Regression (baseline and with class weighting)
  - Model evaluation with confusion matrix, classification report, ROC-AUC

---

## 🔍 Key Results

| Metric                   | Standard LR | Balanced LR |
|--------------------------|:-----------:|:-----------:|
| **Accuracy**             | 0.81        | 0.70        |
| **Precision (Default)**  | 0.68        | 0.39        |
| **Recall (Default)**     | 0.23        | 0.63        |
| **F1 Score (Default)**   | 0.35        | 0.48        |
| **ROC-AUC**              | 0.73        | 0.73        |

- **Class-balanced model**: Recall for defaulters improved from 23% to 63%, with no loss in ROC-AUC.

---

## 🖼️ Key Visualizations

### ROC Curve – Standard Model
![ROC Curve - Standard](figures/roc_curve_standard.png)

### ROC Curve – Balanced Model
![ROC Curve - Balanced](figures/roc_curve_balanced.png)

### Correlation Heatmap
![Correlation Heatmap](figures/corr_heatmap.png)

---

## 📌 Project Highlights

- **Demonstrates**: Handling class imbalance using `class_weight='balanced'`
- **Real-world Relevance**: Credit risk modeling, imbalanced data solutions
- **Well-documented**: Every step from EDA to model interpretation is included in the code and report

---

## 📂 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/credit-default-logreg.git
   cd credit-default-logreg
