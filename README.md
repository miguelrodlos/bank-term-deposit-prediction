# Bank Term Deposit Subscription Prediction

## Overview
This project develops a machine learning pipeline to predict whether a bank client will subscribe to a **term deposit**, using demographic, financial, and interaction data.

The work follows a **complete and reproducible ML workflow**, including preprocessing, model selection, hyperparameter optimization, probability calibration, and final evaluation using a strict outer–inner validation strategy.

## Business Context
Banks aim to optimize marketing campaigns by targeting customers with a higher likelihood of accepting financial products. Accurate predictions allow:
- More efficient client targeting
- Reduced campaign costs
- Improved conversion rates

## Dataset
The dataset contains customer information such as:
- Demographics (age, job, education, marital status)
- Financial indicators (balance, loans, defaults)
- Marketing interaction history (contact type, duration, previous campaigns)

**Target variable**
- `deposit`: whether the client subscribed to a term deposit (binary classification)

> The dataset was provided for academic purposes and is not publicly distributed.

## Methodology

### 1. Exploratory Data Analysis
- Identification of numerical and categorical features
- Detection of missing values, constant columns, and high-cardinality variables
- Careful preprocessing of the `pdays` variable

### 2. Evaluation Design
- **Outer evaluation:** holdout train/test split to estimate future performance
- **Inner evaluation:** cross-validation for model comparison and hyperparameter optimization
- Clear separation between model selection and final testing

### 3. Baseline Models
- Dummy classifier
- k-Nearest Neighbors (KNN)
- Decision Trees (with shallow trees for interpretability)

### 4. Advanced Models
- **Bagging:** Random Forests / Extra Trees
- **Boosting:** Gradient Boosting–based methods
- Comparison of default vs tuned hyperparameters
- Hyperparameter optimization using **Grid Search, Random Search, and Optuna**

### 5. Model Selection and Final Results
- Comparison of all candidates using inner evaluation
- Selection of the best-performing model
- Final evaluation on the test set with confidence intervals
- Training of the final model and generation of competition predictions

### 6. Probability Calibration
- Visual assessment of probability calibration
- Post-hoc calibration methods applied
- Verification that calibration improves probability estimates without degrading performance

## Technologies Used
- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, optuna
- **Methods:** classification, cross-validation, hyperparameter optimization, ensemble learning, probability calibration

## Repository Structure
```text
.
bank-term-deposit-prediction/
├── README.md
├── notebooks/
│   └── training.ipynb
├── models/
│   └── final_model.joblib
├── predictions/
│   └── competition_predictions.csv
├── data/
│   └── README.md
├── requirements.txt
└── .gitattributes
