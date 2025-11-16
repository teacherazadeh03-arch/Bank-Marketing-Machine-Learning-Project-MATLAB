# Bank-Marketing-Machine-Learning-Project-MATLAB
Bank Marketing classification project implemented in MATLAB Online. The project loads the dataset directly from the UCI Machine Learning Repository, preprocesses it, trains multiple machine‑learning models, addresses class imbalance issues, and evaluates model performance using accuracy, confusion matrices, ROC curves, and feature importance.
The goal of this project is to predict whether a customer will subscribe to a bank term deposit based on demographic information and details about previous marketing interactions. This is a binary classification problem (yes or no).
About the Bank Marketing Dataset (UCI Machine Learning Repository)
The Bank Marketing Dataset comes from a series of direct marketing campaigns carried out by a Portuguese retail bank. The goal of the campaign was to encourage customers to subscribe to a term deposit through phone-based outreach.
It is one of the most widely used datasets for classification, predictive modelling, and marketing analytics.
What the Dataset Contains
The dataset includes:
41,188 customer records
21 features (predictors) describing customer demographics, financial behaviour, and details about the marketing campaign
1 target variable, indicating whether the customer subscribed to a term deposit

MATLAB Online–Friendly Implementation

This project is designed specifically for MATLAB Online, meaning:
MATLAB Online cannot interact with local folders easily. To overcome this:
The UCI dataset is downloaded programmatically using websave
The ZIP file is extracted inside the user workspace using unzip
The CSV is then loaded with readtable

Class imbalance handling

The Bank Marketing dataset is heavily imbalanced (~90% "no"). To improve the model:
Logistic regression is trained with class weights, giving more importance to the minority class
Random Forest is used because it is naturally robust to class imbalance
These changes significantly improve predictions for the minority class.

Machine Learning Models Implemented

The project trains three different models:
1. Balanced Logistic Regression
Uses weighted training so minority class contributes more to the loss function
Good baseline model
Outputs probabilities used in ROC/AUC evaluation

2. Decision Tree Classifier
Simple and interpretable
Learns non‑linear decision boundaries
More sensitive to the minority class than unbalanced logistic regression

3. Random Forest (TreeBagger)
Ensemble of 100 decision trees
Handles imbalanced data and noisy features well
Provides feature importance
Typically the best overall performer
