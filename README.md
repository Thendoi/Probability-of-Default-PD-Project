# Probability-of-Default-PD-Project

## Overview

This project predicts credit card default risk using machine learning models and turns model probabilities into business-oriented credit risk decisions.

The objective is to predict whether a borrower will default on a loan using demographic, employment and loan-related characteristics.

The main goal is not only to classify customers as default / non-default, but also to:

compare several machine learning models,
estimate reliable default probabilities,
select a cost-sensitive decision threshold,
explain model predictions using SHAP,
translate model results into practical credit risk recommendations.

## Dataset

The dataset contains:

Borrower demographics (Age, Income, Employment, etc.)
Loan details (Loan Amount, Interest Rate, Loan Term, etc.)
Other relevant variables (Education, Loan Purpose, Marital Status, etc.)


| Variable | Description |
|-----------|-------------|
| person_age | Age of borrower |
| person_income | Annual income |
| person_home_ownership | Home ownership status |
| person_emp_length | Years employed |
| loan_intent | Purpose of loan |
| loan_grade | Loan risk grade |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate |
| loan_status | Default indicator (Target) |
| loan_percent_income | Debt-to-income ratio |
| cb_person_default_on_file | Previous default history |
| cb_person_cred_hist_length | Credit history length |

## Workflow

### 1. Data Exploration

- Examine data types
- Review summary statistics
- Identify missing values
- Detect duplicate records
- Investigate variable distributions

### 2. Data Cleaning

- Remove unrealistic ages
- Remove unrealistic employment lengths
- Handle missing values
- Remove duplicate observations

### 3. Feature Engineering

Categorical variables are transformed using one-hot encoding.

### 4. Model Development

Two models are developed:

#### Logistic Regression

Advantages:

- Interpretable
- Fast
- Common benchmark model

#### XGBoost

Advantages:

- Captures nonlinear relationships
- Handles complex interactions
- Often provides better predictive accuracy

### 5. Model Evaluation

Models are evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC AUC

### 6. Probability of Default (PD)

The final model produces an estimated default probability:

\[
PD = P(Default = 1)
\]

Example:

| Borrower | Predicted PD |
|-----------|-------------|
| Customer A | 2.3% |
| Customer B | 12.8% |
| Customer C | 45.6% |

Higher PD values indicate higher credit risk.

## Applications

- Credit scoring
- Loan approvals
- Risk-based pricing
- Capital modelling
- Portfolio risk management
