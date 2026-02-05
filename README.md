# Bank Term Deposit Subscription Prediction

## Overview
This project develops and evaluates machine learning models to predict whether a bank client will subscribe to a **term deposit** based on demographic, financial, and interaction data.

The work follows a **complete, reproducible ML workflow**, including preprocessing, model selection, hyperparameter optimization, probability calibration, and final evaluation using a strict train/test (outer–inner) framework.

## Business Motivation
Banks aim to optimize marketing campaigns by identifying customers with a higher likelihood of accepting financial products. Accurate predictions allow:
- More efficient targeting
- Reduced campaign costs
- Improved customer experience

## Dataset
The dataset contains customer information such as:
- Demographics (age, job, education, marital status)
- Financial indicators (balance, loans, defaults)
- Marketing interaction history (contact type, duration, previous campaigns)

The target variable is:
- **`deposit`** — whether the client subscribed to a term deposit (binary classification)

> The dataset was provided for academic purposes and is not publicly distributed.

## Methodology

### 1. Exploratory Data Analysis
- Identification of categorical vs numerical features
- Detection of missing values, constant columns, and high-cardinality variables
- Special preprocessing of the `pdays` variable

### 2. Evaluation Design
- **Outer evaluation:** holdout train/test split to estimate future performance
- **Inner evaluation:** cross-validation for model comparison and hyperparameter optimization
- Strict separation between model selection and final evaluation

### 3. Baseline Models
- Dummy classifier
- k-Nearest Neighbors (KNN)
- Decision Trees (shallow trees for interpretability)

### 4. Advanced Models
- **Bagging:** Random Forest / Extra Trees
- **Boosting:** Gradient Boosting family
- Comparison of default vs optimized hyperparameters
- Hyperparameter optimization using Grid/Random Search and **Optuna**

### 5. Model Selection and Final Results
- Inner-evaluation comparison across all candidates
- Selection of the best model
- Final evaluation on the test set with confidence intervals
- Training of the final model and generation of competition predictions

### 6. Probability Calibration
- Visual inspection of calibration curves
- Application of post-hoc calibration techniques
- Verification that calibration improves probability estimates without degrading performance

### 7. Open Choice Task
- Additional modeling or methodological improvement motivated by empirical results

## Technologies Used
- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, optuna
- **Methods:** classification, cross-validation, HPO, ensemble learning, probability calibration

## Repository Structure
```text
├── notebooks/        Jupyter notebook with full analysis
├── models/           Trained final model
├── predictions/      Model predictions on competition data
├── data/             Dataset description (no raw data)
└── README.md
