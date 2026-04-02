# Boosting Techniques: AdaBoost and Gradient Boosting 

This repository provides a hands-on implementation of Boosting algorithms using Scikit-Learn. It covers two primary areas: adaptive boosting for classification tasks and gradient-based boosting for regression, with a focus on model evaluation and optimization.

## Project Overview
Boosting is a powerful ensemble method that builds strong learners from a sequence of weak learners. This project implements:
1. **AdaBoost Classification**: Training an adaptive boosting ensemble to predict liver disease while accounting for dataset imbalance.
2. **Gradient Boosting Regression (GBR)**: Utilizing gradient descent-based boosting to minimize residual errors in continuous value prediction.
3. **Stochastic Gradient Boosting (SGBR)**: Optimizing GBR by introducing stochastic elements (subsampling) to improve generalization and reduce variance.

## Key Features
* **Adaptive Learning**: Implementation of `AdaBoostClassifier` with Decision Trees as base estimators.
* **Imbalanced Data Handling**: Utilizing **ROC AUC** as the primary evaluation metric for classification to ensure performance beyond simple accuracy.
* **Stochastic Optimization**: Fine-tuning Gradient Boosting Regressors using `subsample`, `max_features`, and `n_estimators` to find the optimal bias-variance balance.
* **Metric Benchmarking**: Comprehensive evaluation using **RMSE** (Root Mean Squared Error) for regression tasks.

## Tech Stack
* **Language**: Python
* **Machine Learning**: `scikit-learn` (AdaBoostClassifier, GradientBoostingRegressor)
* **Data Handling**: `pandas`, `numpy`

## Datasets
* **Indian Liver Patient Dataset**: Used for the classification task to predict liver disease based on 10 clinical features.
* **Regression Dataset**: Applied to predict continuous targets Bike Rentals

## Setup & Installation

### 1. Environment Configuration
It is recommended to use a Conda or virtual environment:

```bash
# Create and activate the environment
git clone git clone https://github.com/Joe-Naz01/boosting.git
cd boosting
jupyter notebook

conda create -n boosting_ml python=3.10 -y
conda activate boosting_ml

# Install dependencies
pip install -r requirements.txt