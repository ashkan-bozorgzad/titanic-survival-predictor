# Titanic Survival Prediction

This project applies machine learning to predict whether passengers survived the Titanic disaster based on the classic Titanic dataset. Multiple machine learning models are evaluated to determine the most accurate predictor of survival, offering a comprehensive demonstration of data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Author](#author)

## Project Overview
The goal of this project is to build a predictive model that accurately classifies passengers as survivors or non-survivors based on features like age, sex, class, and fare. This project provides insights into how different machine learning algorithms perform on binary classification tasks and highlights effective techniques for data preprocessing and model tuning.

## Dataset
The dataset used in this project is the [Titanic dataset](https://www.kaggle.com/c/titanic/data), which includes information on passenger demographics, ticket class, fare price, and more. The dataset is divided into training and testing sets for model development and validation.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/titanic-survival-prediction.git
    cd titanic-survival-prediction
    ```

2. **Install required libraries**:
    This project uses several Python libraries. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    **Main Libraries**:
    - `pandas`, `numpy` for data handling
    - `matplotlib`, `seaborn` for visualization
    - `scikit-learn` for machine learning models and evaluation
    - `xgboost` and `lightgbm` for advanced ensemble models

3. **Download the dataset**:
   Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place it in the project directory.

## Project Workflow
1. **Data Exploration**: Analyze the dataset to understand its structure, visualize distributions, and identify missing values.
2. **Data Preprocessing**:
   - Handle missing values through imputation.
   - Encode categorical variables.
   - Scale numerical features to standardize the input data.
3. **Feature Engineering**: Apply feature selection techniques to retain the most relevant features.
4. **Model Training**: Train several models, including Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
5. **Model Evaluation**: Use cross-validation and grid search for hyperparameter tuning, selecting the best model based on accuracy and computational efficiency.

## Model Evaluation
Each model is evaluated based on accuracy using cross-validation and hyperparameter tuning with GridSearchCV to improve model performance. The results from each model are compared to determine the most effective predictor of survival.

## Results
The selected model is tested on the test set to assess its generalization ability. Results indicate that ensemble models, such as Random Forest and XGBoost, provide high accuracy in predicting Titanic survival, outperforming simpler models like Logistic Regression.

## Author
This project was developed by **Ashkan Bozorgzad** as part of a machine learning study. The goal was to explore various machine learning techniques and gain insights into their performance on binary classification tasks.

---

Feel free to reach out with any questions or feedback!
