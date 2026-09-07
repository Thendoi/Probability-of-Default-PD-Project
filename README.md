# Probability of Default (PD) Model

## Project Overview:

This project develops an end-to-end **Probability of Default (PD) modelling framework** to predict whether a borrower is likely to default on a loan and, more importantly, translate those predictions into actionable lending decisions.

Using approximately **32,500 loan applications**, the project focuses not only on predicting default but also on understanding whether the models can effectively **rank borrowers according to their level of credit risk**.

## Business Problem:

Lenders face a fundamental trade-off when approving loans. Approving a high-risk borrower can result in financial losses, while declining a low-risk borrower can result in lost business opportunities.

For example, instead of making a simple approve/decline decision, a lender could estimate that an applicant has an **8% probability of default** and use that information to support a more informed decision.

The primary objectives of this project are:

- Predict the probability that a borrower will default
- Identify important drivers of credit risk
- Rank borrowers according to their level of default risk
- Segment borrowers into different risk categories

By converting borrower characteristics into an estimated **Probability of Default**, lenders can make more consistent and risk-sensitive decisions while balancing **credit risk, customer growth, and profitability**.

## The dataset
 
Loan applications with borrower demographics, employment info, and loan details:
 
| Variable | Description |
|-----------|-------------|
| `person_age` | Age of borrower |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Years employed |
| `loan_intent` | Purpose of loan |
| `loan_grade` | Loan risk grade (A–G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_status` | Target — 1 = default, 0 = repaid |
| `loan_percent_income` | Debt-to-income ratio |
| `cb_person_default_on_file` | Previous default history |
| `cb_person_cred_hist_length` | Credit history length |

## Project Methodology

### Phase 1: Data Understanding

- Dataset inspection
- Target variable analysis
- Data type review
- Class imbalance evaluation

The target variable was imbalanced, with approximately **22% of borrowers defaulting**.

### Phase 2: Data Cleaning

- Identified and removed observations with impossible ages
- Removed observations with unrealistic employment lengths
- Imputed missing interest rates using the median
- Checked for duplicate observations
- Performed data consistency checks

### Phase 3: Feature Engineering & Encoding

The project included both **application and behavioural variables** to capture different dimensions of borrower risk.

Key preprocessing steps included:

- Retained `loan_grade` as an ordinal variable
- One-hot encoded nominal categorical variables
- Encoded home ownership categories
- Encoded loan intent categories
- Prepared numerical variables for model development

Keeping `loan_grade` was particularly important because it captures an ordered measure of borrower credit quality.

### Phase 4: Data Splitting & Class Imbalance

The data was first divided into **training and testing datasets** before addressing class imbalance.

**SMOTE (Synthetic Minority Over-sampling Technique)** was then applied only to the training data.

This approach prevents synthetic observations generated from the training data from leaking into the test set and producing overly optimistic model performance.

### Phase 5: Model Development

Three machine learning models were developed and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

Logistic Regression was used as an **interpretable baseline**, while Random Forest and XGBoost were used to capture more complex non-linear relationships and interactions between borrower characteristics.

### Phase 6: Model Evaluation

# Model Performance

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1** | **ROC-AUC** | **KS** |
|-----------|--------------|---------------|------------|--------|-------------|--------|
| Logistic Regression | 0.804 | 0.532 | 0.780 | 0.633 | 0.867 | 0.592 |
| Random Forest | 0.932 | 0.924 | 0.744 | 0.825 | 0.938 | 0.738 |
| XGBoost | 0.936 | 0.942 | 0.752 | 0.836 | 0.947 | 0.761 |

### Best Performing Model

**XGBoost**

XGBoost delivered the strongest overall performance, achieving the highest **ROC-AUC and KS statistic** while maintaining a strong balance between precision and recall.

The KS statistic of **0.761** indicates strong discriminatory power between defaulting and non-defaulting borrowers.

## Risk Ranking & Decile Analysis

Model performance was also evaluated by ranking borrowers according to their predicted Probability of Default (PD).

The test population was divided into **10 risk deciles**, from the safest borrowers to the riskiest borrowers.

| **Decile** | **Actual Default Rate** |
|------------|--------------------------|
| 1 (Safest) | 0.0% |
| 2 | 0.8% |
| 3 | 1.7% |
| 4 | 3.7% |
| 5 | 4.1% |
| 6 | 6.2% |
| 7 | 12.2% |
| 8 | 16.5% |
| 9 | 70.6% |
| 10 (Riskiest) | 100.0% |

The strong increase in observed default rates across the risk deciles demonstrates that the model is effectively **ranking borrowers from lower to higher credit risk**.

The sharp increase in default rates in the highest-risk deciles provides a potentially useful basis for setting underwriting cut-offs and identifying borrowers requiring additional review.

## Probability of Default in Practice

The model produces an estimated **Probability of Default (PD)** for each borrower:

$$
PD = P(Default = 1)
$$

For example:

| **Borrower** | **Predicted PD** | **Risk Tier** |
|--------------|------------------|---------------|
| Customer A | 2.3% | Low |
| Customer B | 12.8% | Medium |
| Customer C | 45.6% | High |

Instead of producing only a binary approve/decline prediction, the model provides a probability that can be used to support **risk-based lending decisions**.

## Key Findings

### Feature Importance

The strongest predictors across the models included:

- `loan_grade`
- `loan_percent_income`
- `loan_int_rate`

These variables were consistently important across the three models and align with fundamental credit-risk principles.

`loan_grade` captures borrower credit quality, while `loan_percent_income` provides an indication of the borrower's debt burden relative to income. Interest rate can also reflect the level of risk associated with the loan.

### Model Comparison

The tree-based models, particularly **XGBoost**, outperformed Logistic Regression in overall discriminatory power.

This suggests that the relationship between borrower characteristics and default risk is not purely linear. Tree-based models can capture **non-linear relationships and interactions**, such as how income, loan amount, interest rate, and credit grade interact to influence default risk.

### Interpretability

Although Logistic Regression produced lower predictive performance than the tree-based models, it remains valuable in credit risk because of its **interpretability**.

The model coefficients can provide a clearer explanation of how individual variables influence predicted default risk, which is important when models need to be understood, challenged, or explained to business stakeholders and regulators.

## Business Application

The PD model can support several practical credit-risk decisions:

### Low-Risk Borrowers

- Faster approval
- Competitive pricing
- Potentially higher credit limits

### Medium-Risk Borrowers

- Conditional approval
- More conservative exposure limits
- Additional monitoring

### High-Risk Borrowers

- Manual credit review
- Additional documentation
- More conservative lending terms

### Very High-Risk Borrowers

- Decline application
- Avoid excessive exposure
- Reduce concentration of high-risk borrowers

The final decision threshold should ultimately be determined using the **economic cost of false positives and false negatives**, rather than relying solely on a statistical metric such as F1 Score.

## What I Learned

- Credit risk models should be evaluated on **discrimination and ranking**, not accuracy alone.
- Applying SMOTE before the train/test split can introduce **data leakage** and produce misleading performance.
- `loan_grade`, `loan_percent_income`, and `loan_int_rate` provided strong and economically intuitive signals of default risk.
- Risk-decile analysis provides a practical way to determine whether a model can meaningfully separate low-risk and high-risk borrowers.

## Limitation or recommendations

The natural next steps for the project include:

- Add **SHAP** for individual-level model explanations
- Determine decision thresholds using the economic cost of false positives and false negatives
- Perform probability calibration for applications involving pricing or expected loss estimation
- Hyperparameter tuning for Random Forest and XGBoost
- Conduct out-of-time validation using a more recent sample
