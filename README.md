# Probability of Default (PD) Model

## Project Overview:

This project develops an end-to-end **Probability of Default (PD) modelling framework** to predict whether a borrower is likely to default on a loan and, more importantly, translate those predictions into actionable lending decisions.

Using approximately **32,500 loan applications**, the project trains and compares three machine learning models: **Logistic Regression, Random Forest, and XGBoost**.

The project focuses not only on predicting default but also on understanding whether the models can effectively **rank borrowers according to their level of credit risk**. This is particularly important in real-world lending, where risk ranking can support approval decisions, risk-based pricing, portfolio monitoring, and provisioning.

The project follows a practical credit risk modelling workflow, including data preparation, class imbalance treatment, model development, model evaluation, and risk segmentation.


## Business Problem:

Lenders face a fundamental trade-off when approving loans. Approving a high-risk borrower can result in financial losses, while declining a low-risk borrower can result in lost business opportunities.

Traditional lending decisions may rely on fixed rules or credit score thresholds, which can fail to capture the full range of borrower risk.

A Probability of Default model makes this risk more explicit by estimating the likelihood that a borrower will default. For example, instead of making a simple approve/decline decision, a lender could estimate that an applicant has an **8% probability of default** and use that information to support a more informed decision.

The primary objectives of this project are:

- Predict the probability that a borrower will default
- Identify important drivers of credit risk
- Rank borrowers according to their level of default risk
- Segment borrowers into different risk categories
- Support data-driven underwriting decisions
- Enable risk-based pricing and approval cut-offs
- Identify high-risk borrowers for additional review
- Support portfolio risk management
- Provide inputs that can support provisioning and expected credit loss analysis

By converting borrower characteristics into an estimated **Probability of Default**, lenders can make more consistent and risk-sensitive decisions while balancing **credit risk, customer growth, and profitability**.










## What's in this project

I trained and compared three models on ~32,500 loan applications: Logistic Regression, Random Forest, and XGBoost. 

- **Fixed a data leakage issue.** The naive approach is to balance the classes with SMOTE and *then* split into train/test. That lets synthetic training rows leak into the test set and makes your accuracy look better than it really is. I split first, then balanced only the training data.
- **Kept `loan_grade` instead of dropping it.** It's an ordinal risk grade (A through G), and it turned out to be one of the strongest predictors — throwing it away would've cost real signal.
- **Evaluated it the way a credit risk person actually would**, not just accuracy. Confusion matrices, ROC/AUC, and the KS statistic (the standard way to measure how well a scorecard separates good and bad borrowers).
- **Checked that the model actually ranks risk**, using a risk-decile breakdown — because a model that's 90% accurate but can't tell your safest customers from your riskiest ones isn't actually useful.

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

## How I got there

1. **Explored the data** — checked distributions, missing values, and found the target is imbalanced (about 22% default rate).
2. **Cleaned it** — removed a handful of rows with impossible ages and employment lengths, imputed missing interest rates with the median.
3. **Encoded features** — kept `loan_grade` as an ordinal number, one-hot encoded the nominal categories (home ownership, loan intent).
4. **Split before balancing** — train/test split first, then SMOTE on the training set only.
5. **Trained three models** — Logistic Regression as an interpretable baseline, then Random Forest and XGBoost to see how much non-linear patterns actually help.
6. **Evaluated properly** — confusion matrix, ROC/AUC, KS statistic, and a risk-decile gains chart.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC | KS |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.532 | 0.780 | 0.633 | 0.867 | 0.592 |
| Random Forest | 0.932 | 0.924 | 0.744 | 0.825 | 0.938 | 0.738 |
| XGBoost | 0.936 | 0.942 | 0.752 | 0.836 | 0.947 | 0.761 |

XGBoost came out on top, with a KS of 0.76 — a KS above roughly 0.4 is generally considered solid for a credit scorecard, so this is a strong result.

The part I actually care about more than the metrics table, though, is whether the model ranks risk sensibly. So I split the test set into 10 deciles by predicted probability and looked at the actual default rate in each:

| Decile | Actual default rate |
|---|---|
| 1 (safest) | 0.0% |
| 2 | 0.8% |
| 3 | 1.7% |
| 4 | 3.7% |
| 5 | 4.1% |
| 6 | 6.2% |
| 7 | 12.2% |
| 8 | 16.5% |
| 9 | 70.6% |
| 10 (riskiest) | 100.0% |

That jump at decile 9 and 10 is the interesting part. It means declining just the riskiest 10% of applicants would catch the large majority of expected defaults while barely affecting the other 90% of approvals — which is the kind of trade-off a credit team can actually act on, not just a number on a slide.

### What a prediction looks like in practice

$$PD = P(\text{Default} = 1)$$

| Borrower | Predicted PD | Risk tier |
|-----------|-------------|-----------|
| Customer A | 2.3% | Low |
| Customer B | 12.8% | Medium |
| Customer C | 45.6% | High |

## What I learned / noticed

- `loan_grade`, `loan_percent_income`, and `loan_int_rate` were consistently the top predictors across all three models — which matches what you'd expect from credit risk fundamentals, and is a nice sanity check that the model isn't picking up on noise.
- The tree-based models beat Logistic Regression on ranking power, mostly because they pick up on interactions (e.g. how income and loan amount together affect risk) that a linear model can't.
- Logistic Regression still has a place though — not because it wins on the numbers, but because you can read off exactly why it made a decision, which matters a lot if you ever need to explain a decline to a regulator or a customer.

## Where this could go next

I haven't done these yet, but they're the natural next steps if I keep working on this:

- Add SHAP so individual predictions can be explained row-by-row (useful for adverse-action notices)
- Pick a decision threshold based on actual cost of a false positive vs. false negative, instead of just optimizing F1
- Calibrate the probabilities properly if they're going to be used for pricing, not just ranking
- Tune hyperparameters on Random Forest and XGBoost rather than using defaults
- Build a small Streamlit app so someone could try scoring a new applicant without touching the notebook
- Test on a more recent, out-of-time sample before I'd trust this in production

