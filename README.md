#  Bank Fraud Detection using Machine Learning

An end-to-end Machine Learning project that explores the complete workflow of building a fraud detection system—from data exploration and feature engineering to preprocessing, model training, and evaluation.

---

##  Table of Contents

* [Project Overview](#-project-overview)
* [Objectives](#-objectives)
* [Project Structure](#-project-structure)
* [Dataset Summary](#-dataset-summary)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [SQL Analysis](#-sql-analysis)
* [Feature Engineering](#-feature-engineering)
* [Data Preprocessing](#-data-preprocessing)
* [Handling Class Imbalance](#-handling-class-imbalance)
* [Machine Learning Models](#-machine-learning-models)
* [Evaluation Metrics](#-evaluation-metrics)
* [Experimental Findings](#-experimental-findings)
* [Key Learnings](#-key-learnings)
* [Technologies Used](#-technologies-used)
* [Future Improvements](#-future-improvements)
* [Conclusion](#-conclusion)

---

#  Project Overview

Fraud detection is one of the most important applications of Machine Learning in the banking and financial services industry. Financial institutions process millions of transactions every day, making it impossible to manually identify fraudulent activities. Machine learning models can assist by learning transaction patterns and flagging potentially fraudulent transactions for further investigation.

This project demonstrates a complete Machine Learning workflow for fraud detection, starting from exploratory data analysis and SQL-based business analysis to preprocessing, feature engineering, model training, and evaluation.

The primary objective of this project was **not only to build classification models**, but also to follow a structured and production-oriented Machine Learning pipeline while understanding the challenges associated with highly imbalanced datasets.

---

#  Objectives

The project aims to:

* Perform comprehensive Exploratory Data Analysis (EDA).
* Analyze the dataset using SQL to extract business insights.
* Engineer meaningful features from raw transaction data.
* Build a reusable preprocessing pipeline using `Pipeline` and `ColumnTransformer`.
* Handle severe class imbalance using different techniques.
* Train multiple Machine Learning models.
* Compare model performance using appropriate evaluation metrics.
* Draw conclusions based on experimental evidence rather than assumptions.

---

#  Project Structure

```text
Bank-Fraud-Detection/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── artifacts/
│   ├── preprocessor.pkl
│   ├── X_train_processed.pkl
│   ├── X_test_processed.pkl
│   ├── X_train_balanced.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── y_train_balanced.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── sql_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_building.ipynb
│
└── app.py
```

---

#  Dataset Summary

The dataset consists of banking transaction records containing customer information, transaction details, account information, merchant details, and a fraud label.

### Dataset Characteristics

| Property                | Value        |
| ----------------------- | ------------ |
| Number of Records       | **200,000**  |
| Total Features          | **24**       |
| Target Variable         | **Is_Fraud** |
| Fraudulent Transactions | **5.04%**    |
| Legitimate Transactions | **94.96%**   |

The dataset is highly imbalanced, making **Accuracy** an unreliable metric for evaluating model performance.

---

#  Exploratory Data Analysis

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

* No missing values were found.
* No duplicate records were present.
* Numerical features did not contain significant outliers.
* Fraudulent transactions represented only **5.04%** of the dataset.
* Numerical variables exhibited weak linear correlations.
* Several identifier and low-information columns were removed before model training.

---

#  SQL Analysis

In addition to EDA, SQL was used to analyze the dataset from a business perspective.

The analysis included:

* Fraud rates across different states
* Merchant category analysis
* Device-wise fraud distribution
* Account type analysis
* Transaction channel analysis
* Age group analysis
* Gender-wise fraud analysis

These queries helped uncover business insights that complemented the findings from exploratory data analysis.

---

#  Feature Engineering

Several new features were created to better represent transaction behavior.

### 1. Day of Week

Extracted from `Transaction_Date` to capture weekday and weekend transaction patterns.

### 2. Hour

Extracted from `Transaction_Time` to capture time-of-day transaction behavior.

### 3. Amount Balance Ratio

```text
Transaction_Amount / Account_Balance
```

This feature represents the proportion of the customer's account balance involved in a transaction.

---

#  Data Preprocessing

A reusable preprocessing pipeline was implemented using Scikit-learn.

### Preprocessing Steps

* Removed identifier columns
* Removed redundant features
* Performed feature engineering
* Stratified Train-Test Split
* Standardized numerical features using `StandardScaler`
* One-Hot Encoded categorical features using `OneHotEncoder`
* Combined transformations using `ColumnTransformer`

The fitted preprocessing pipeline was serialized using **Joblib** for future inference.

---

#  Handling Class Imbalance

The dataset contained only **5.04%** fraudulent transactions.

To address this challenge, multiple imbalance handling strategies were explored.

### Baseline

Models trained directly on the original dataset.

### SMOTE

Synthetic Minority Oversampling Technique (SMOTE) was applied **only to the training set** to avoid data leakage.

### Class Weights

For supported algorithms, balanced class weights were used to penalize misclassification of the minority class.

This allowed comparison of different imbalance handling strategies under the same evaluation framework.

---

#  Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Logistic Regression + SMOTE
* Decision Tree
* Decision Tree + SMOTE
* Random Forest
* Random Forest + Class Weights

Each model was evaluated on the untouched test dataset.

---

#  Evaluation Metrics

The following evaluation metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Given the highly imbalanced nature of the dataset, greater importance was placed on **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** rather than overall Accuracy.

---

#  Experimental Findings

The experiments revealed the following observations:

* Logistic Regression trained on the original dataset predicted almost every transaction as legitimate.
* Applying SMOTE improved recall for Logistic Regression but significantly reduced precision.
* Decision Tree models showed only marginal predictive performance.
* Random Forest models did not exhibit meaningful improvement after applying class weighting.
* Across all experiments, the ROC-AUC remained close to **0.50**, indicating poor class separability.

These findings suggest that the available features contain limited predictive information for distinguishing fraudulent transactions.

---

#  Key Learnings

One of the most valuable outcomes of this project is understanding that **a well-designed Machine Learning pipeline does not always guarantee a high-performing model**.

This project reinforced several important Machine Learning principles:

* Data quality is as important as model selection.
* Preventing data leakage is critical.
* Class imbalance should be handled carefully and evaluated experimentally.
* Accuracy alone is not an appropriate metric for imbalanced classification problems.
* Honest interpretation of model performance is more valuable than reporting artificially inflated metrics.
* Systematic experimentation is an essential part of building reliable Machine Learning systems.

---

#  Technologies Used

| Category             | Technologies             |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Data Manipulation    | Pandas, NumPy            |
| Visualization        | Matplotlib, Seaborn      |
| Machine Learning     | Scikit-learn             |
| Imbalance Handling   | Imbalanced-learn (SMOTE) |
| Model Serialization  | Joblib                   |
| Database Analysis    | SQL                      |

---

#  Future Improvements

Potential enhancements include:

* Evaluate advanced ensemble methods such as XGBoost, LightGBM, and CatBoost.
* Perform hyperparameter tuning using Grid Search or Randomized Search.
* Implement explainable AI techniques using SHAP.
* Deploy the trained model using Streamlit or Flask.
* Engineer additional behavioral and temporal features.
* Evaluate the pipeline on a real-world fraud detection dataset with stronger predictive signals.

---

#  Conclusion

This project demonstrates a complete supervised Machine Learning workflow for fraud detection, covering data exploration, SQL-based business analysis, feature engineering, preprocessing, class imbalance handling, model training, and evaluation.

Although the evaluated models did not achieve strong predictive performance, the project highlights the importance of systematic experimentation, rigorous evaluation, and evidence-based decision-making. The consistently low ROC-AUC scores suggest that the available features provide limited predictive signal for the target variable.

More importantly, this project demonstrates the ability to build a structured Machine Learning pipeline, apply best practices to avoid data leakage, compare multiple modeling strategies, and critically interpret experimental results—skills that are directly applicable to real-world Machine Learning projects.
