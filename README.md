# Liquidity Prediction & Capital Funding Advisory

A machine-learning case study on moving from static peer analysis toward predictive liquidity planning for Fortune 500 finance teams.

## Overview

A capital-funding advisory team wanted to help CFOs and treasurers answer:

- How much liquidity should a company maintain?
- How does its liquidity position compare with peers?

The existing analysis used simple linear regression on publicly available competitor data and produced a reported **0.48%** result. The team wanted to assess whether machine learning could provide a stronger predictive foundation and eventually support **3-month, 6-month, and 9-month liquidity planning**.

## Approach

I developed a regression modeling framework in **Python** using **scikit-learn**.

- Set available liquidity as the target variable.
- Split the data into **80% training** and **20% held-out test** sets, using a fixed random state.
- Reviewed descriptive statistics, the target distribution, and missing values.
- Evaluated five regression models:
  - Lasso Regression
  - Ridge Regression
  - Elastic Net
  - Random Forest
  - Gradient Boosting
- Used model pipelines and **GridSearchCV with 5-fold cross-validation** for hyperparameter tuning.
- Assessed performance on the held-out test set using **R²** and **Mean Absolute Error (MAE)**.

## Results

The Gradient Boosting model achieved:

| Metric | Test result |
|---|---:|
| R² | 0.884 |
| MAE | 532.453 |

The model explained approximately **88.4% of the variation** in available liquidity within the test dataset. MAE represents the typical absolute prediction error in the target variable’s units.

> **Note:** The original analysis’s reported 0.48% and the model’s R² and MAE are different measures. They should not be described as a direct percentage-to-percentage improvement without confirming what the 0.48% statistic represents.

## Advisory Application

The model’s intended role is to support—not replace—financial judgment. A potential workflow is:

**Current liquidity → Peer comparison → Predicted liquidity → Funding considerations → Executive recommendation**

Predictions could inform planning across **3-, 6-, and 9-month horizons**. Business and market factors not represented in the dataset would still need to be considered.

## Potential Executive Deliverable

A presentation for CFOs and treasurers could include:

- Current liquidity position and relevant peer comparisons
- Forward-looking liquidity expectations
- Potential capital-funding implications
- 3-month, 6-month, and 9-month planning considerations
- Key model findings and limitations
- Additional considerations before making a funding decision

## Business Value

The framework provides a more systematic, predictive approach than relying on a single linear-regression specification. It supports:

- Comparison of multiple modeling approaches
- Cross-validation and hyperparameter tuning
- Evaluation on previously unseen data
- Quantified model performance using R² and MAE
- A foundation for repeatable, forward-looking liquidity analysis
- Clear communication of findings for finance decision-makers

## Tools & Methods

- Python
- scikit-learn
- Regression modeling
- Train/test split
- Grid search and 5-fold cross-validation
- R² and Mean Absolute Error

## Limitations

Predictions are limited by the available data and features. The model does not capture every business, market, or strategic factor relevant to funding decisions. It is a **decision-support tool**, not a substitute for financial judgment.