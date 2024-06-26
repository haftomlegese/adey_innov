# Fraud Detection Model Project

This project focuses on improving fraud detection for e-commerce and bank credit transactions using machine learning models. It includes detailed data analysis, feature engineering, model building, explainability, and deployment as an API using Flask and Docker.


## Project Overview

This project aims to create accurate and robust fraud detection models for e-commerce transactions and bank credit transactions. It leverages data analysis, geolocation analysis, transaction pattern recognition, and advanced machine learning models to improve fraud detection accuracy.

## Business Need

Adey Innovations Inc. focuses on solutions for e-commerce and banking sectors. By improving fraud detection, the company aims to enhance transaction security, prevent financial losses, and build trust with customers and financial institutions. This project involves analyzing transaction data, engineering features, building and training models, evaluating their performance, and deploying the models for real-time detection.

## Project Structure
- `data/`: Directory containing raw and cleaned datasets.
- `notebooks/`: Jupyter notebooks used for analysis and visualization.
- `scripts/`: Python scripts for data processing and visualization.
- `tests/`: Directory for test.
- `fraud_detection`: Directory for Model Deployment and API Development
- `README.md`: Project overview and instructions.
- `requirements.txt`: Python dependencies required for the project.

## Tasks
### Task 1: Data Analysis and Preprocessing
- Handle Missing Values
- Data Cleaning
- Exploratory Data Analysis
    - Univariate analysis.
    - Bivariate analysis.
- Merge Datasets for Geolocation Analysis
    - Convert IP addresses to integer format.
    - Merge Fraud_Data.csv with IpAddress_to_Country.csv.
- Feature Engineering
    - Time-Based features for Fraud_Data.csv:
        - hour_of_day
        - day_of_week
- Normalization and Scaling

### Task 2: Model Building and Training
- Data Preparation
    - Feature and Target Separation [‘Class’(creditcard), ‘class’(Fraud_Data)]
    - Train-Test Split.

- Model Selection:

    - Logistic Regression.
    - Decision Tree.
    - Random Forest.
    - Gradient Boosting.
    - Multi-Layer Perceptron (MLP).
    - Convolutional Neural Network (CNN).
    - Recurrent Neural Network (RNN).
    - Long Short-Term Memory (LSTM).

- Model Training and Evaluation
    - Training models for both credit card and fraud-data datasets.
- MLOps Steps
    - Versioning and Experiment Tracking using tools like MLflow to track experiments, log parameters, metrics, and version models.

### Task 3: Model Explainability
- Using SHAP for Explainability
    - Explaining a Model with SHAP:
        - Summary Plot.
        - Force Plot.
        - Dependence Plot.
- Using LIME for Explainability
    - Explaining a Model with LIME:
        - Feature Importance Plot.

#### N.B from task 1 to task 3 you can find them on notebooks/adey_data_analysis.ipynb 

### Task 4: Model Deployment and API Development
- Setting Up the Flask API
- can be found on the <fraud_detection directory>.
