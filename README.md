# Bank Fraud Detection using Machine Learning

## Project Overview

Fraud detection is one of the most important applications of machine learning in the banking and financial services industry. Financial institutions process millions of transactions every day, making it impossible to manually identify fraudulent activities. Machine learning models can assist by learning transaction patterns and flagging potentially fraudulent transactions for further investigation.

This project demonstrates an end-to-end machine learning workflow for fraud detection, starting from exploratory data analysis to model training and evaluation. The primary objective was not only to build classification models but also to follow a systematic and production-oriented machine learning pipeline while understanding the challenges associated with highly imbalanced datasets.

---

# 🎯 Objectives

The project aims to:

* Perform comprehensive Exploratory Data Analysis (EDA).
* Analyze the dataset using SQL to extract business insights.
* Engineer meaningful features from raw transaction data.
* Build a reusable preprocessing pipeline using `Pipeline` and `ColumnTransformer`.
* Handle severe class imbalance using different techniques.
* Train multiple machine learning models.
* Compare model performance using appropriate evaluation metrics.
* Draw conclusions based on experimental results rather than assumptions.

---

# 📂 Project Structure

```
Bank-Fraud-Detection/
│
├── .gitignore               # Keeps your GitHub repo clean from data/caches
├── README.md                # Project documentation
├── requirements.txt         # Project dependencie
│
├── artifacts/                  # Saved .pkl / .joblib artifacts (ignored by git)
│   ├── preprocessor.pkl
│   └── X_train_processed.pkl
│   └── X_test_processed.pkl
│   └── y_test.pkl
│   └── y_train.pkl
├── notebooks/               # Dedicated space for notebooks
│   ├── eda.ipynb
│   ├── sql_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_building.ipynb # Fixed typo!
│

---

# Dataset Summary

The dataset consists of banking transaction records containing customer information, transaction details, account information, merchant details, and a fraud label.

### Dataset Characteristics

* Number of records: **200,000**
* Total features: **24**
* Target variable: **Is_Fraud**
* Fraudulent transactions: **5.04%**
* Legitimate transactions: **94.96%**

The dataset is highly imbalanced, making accuracy an unreliable metric for model evaluation.

---

# Exploratory Data Analysis

The following analyses were performed:

* Dataset overview
* Missing value analysis
* Duplicate detection
* Numerical feature analysis
* Categorical feature analysis
* Target variable distribution
* Correlation analysis
* Box plot analysis for outlier detection

### Key Findings

* No missing values were present.
* No duplicate records were found.
* Numerical features did not exhibit significant outliers.
* The fraud class constituted only about **5%** of the dataset.
* Numerical features showed weak linear correlations.
* Several columns were identified as identifiers or low-information features and removed prior to modeling.

---

# SQL Analysis

In addition to EDA, SQL was used to explore the dataset from a business perspective.

Examples of analyses include:

* Fraud rates across states
* Fraud rates by merchant category
* Device-wise fraud distribution
* Account type analysis
* Transaction channel analysis
* Age group analysis
* Gender-based fraud analysis

This helped identify business patterns that may not be immediately visible through visualizations alone.

---

# ⚙ Feature Engineering

The following features were created:

### Day of Week

Extracted from:

```
Transaction_Date
```

to capture weekday vs weekend transaction behavior.

### Hour

Extracted from:

```
Transaction_Time
```

to capture time-based transaction patterns.

### Amount Balance Ratio

```
Transaction_Amount / Account_Balance
```

This feature captures the proportion of the customer's balance involved in the transaction.

---

# Data Preprocessing

A complete preprocessing pipeline was implemented using Scikit-learn.

### Steps

* Removed identifier columns
* Removed redundant columns
* Feature engineering
* Train-test split using stratified sampling
* Numerical preprocessing using StandardScaler
* Categorical preprocessing using OneHotEncoder
* Combined preprocessing using ColumnTransformer

The fitted preprocessor was saved using Joblib for future inference.

---

# ⚖ Handling Class Imbalance

Since fraudulent transactions represented only **5%** of the dataset, multiple imbalance handling strategies were explored.

### Baseline

Models trained on the original dataset.

### SMOTE

Synthetic Minority Oversampling Technique was applied only to the training data to avoid data leakage.

### Class Weights

Models supporting class weighting were trained using balanced class weights.

This comparison helped evaluate the effectiveness of different imbalance handling strategies.

---

# Machine Learning Models

The following models were evaluated:

* Logistic Regression
* Logistic Regression + SMOTE
* Decision Tree
* Decision Tree + SMOTE
* Random Forest
* Random Forest + Class Weights

Each model was evaluated on the untouched test set.

---

# Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Given the highly imbalanced nature of the dataset, greater emphasis was placed on Precision, Recall, F1 Score, and ROC-AUC rather than Accuracy.

---

# Experimental Findings

The experiments demonstrated that:

* Logistic Regression trained on the original data predicted almost every transaction as legitimate.
* Applying SMOTE improved recall for Logistic Regression but significantly reduced precision.
* Decision Tree models showed limited predictive performance.
* Random Forest models did not exhibit significant improvement even after class weighting.
* All evaluated models achieved ROC-AUC values close to **0.50**, indicating poor class separability.

---

# Key Learning

One of the most important takeaways from this project is that a well-designed machine learning pipeline does not necessarily produce a high-performing model.

This project reinforces several practical lessons:

* Proper preprocessing is essential.
* Preventing data leakage is critical.
* Multiple imbalance handling techniques should be compared rather than assumed to be beneficial.
* Model performance should always be validated using appropriate evaluation metrics.
* Honest interpretation of experimental results is more valuable than overfitting a model to achieve better scores.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib
* SQL

---

# Future Improvements

Potential improvements include:

* Evaluating boosting algorithms such as XGBoost, LightGBM, and CatBoost.
* Performing hyperparameter tuning.
* Implementing explainability using SHAP.
* Building a Streamlit or Flask application for inference.
* Experimenting with additional behavioral features.
* Evaluating the pipeline on a real-world fraud detection dataset.

---

# Conclusion

This project demonstrates an end-to-end supervised machine learning workflow for fraud detection, including data exploration, SQL-based business analysis, feature engineering, preprocessing, class imbalance handling, model training, and evaluation.

Although the evaluated models did not achieve strong predictive performance, the project highlights the importance of systematic experimentation, rigorous evaluation, and evidence-based conclusions. The experience gained from building a complete machine learning pipeline is directly transferable to real-world classification problems where data quality and feature informativeness play a critical role in model performance.
