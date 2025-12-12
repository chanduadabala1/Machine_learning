
 Telco Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn in a telecommunications dataset. By analyzing various customer attributes and service usage patterns, we aim to build machine learning models that can identify customers at risk of churning, allowing the company to proactively intervene and improve customer retention.

## Dataset
The dataset used is `WA_Fn-UseC_-Telco-Customer-Churn.csv`, which contains information about a fictional telecommunications company's customers, including demographics, services subscribed to, and churn status.

## Methodology
The project follows a standard machine learning pipeline:
1.  **Data Loading**: The `WA_Fn-UseC_-Telco-Customer-Churn.csv` file is loaded into a pandas DataFrame.
2.  **Data Cleaning & Preprocessing**:
    *   `customerID` is dropped as it has no predictive power.
    *   `TotalCharges` is converted to numeric, handling missing values by coercion and dropping rows with nulls.
    *   The `Churn` target variable is encoded (Yes=1, No=0).
    *   Categorical features are one-hot encoded.
3.  **Data Splitting and Scaling**:
    *   The dataset is split into training (80%) and testing (20%) sets.
    *   Numerical features are scaled using `StandardScaler` (important for Logistic Regression).
4.  **Model Training**:
    *   **Logistic Regression**: A baseline model for linear relationships.
    *   **Random Forest Classifier**: A more robust model capable of capturing non-linear relationships.
5.  **Evaluation**:
    *   Models are evaluated using **Accuracy**, **F1 Score**, and **Classification Report**.
    *   Confusion matrices are plotted for both models.
6.  **Feature Importance**:
    *   The top 10 most important features contributing to churn are identified using the Random Forest model and visualized.

## Key Results

### Model Performance:
*   **Logistic Regression**:
    *   Accuracy: 0.7875
    *   F1 Score: 0.5635
*   **Random Forest**:
    *   Accuracy: 0.7854
    *   F1 Score: 0.5410

*(Note: Exact scores may vary slightly depending on `random_state` and specific data splits.)*

Both models show similar performance, with Logistic Regression slightly outperforming Random Forest in terms of F1 Score in this particular run. This suggests that while Random Forest is good at capturing complex relationships, the data might have strong linear components that Logistic Regression handles well, or that further hyperparameter tuning for Random Forest might be beneficial.

### Top Feature Importances (from Random Forest):
1.  **TotalCharges**
2.  **MonthlyCharges**
3.  **tenure**
4.  **InternetService_Fiber optic**
5.  **PaymentMethod_Electronic check**
6.  **OnlineSecurity_Yes**
7.  **Contract_Two year**
8.  **gender_Male**
9.  **TechSupport_Yes**

These features are the most influential in predicting customer churn according to the Random Forest model.

## Setup and Usage

### Prerequisites
*   Python 3.x
*   Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
1.  **Download the dataset**: Obtain the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file and place it in the same directory as your notebook or upload it to your Colab environment.
2.  **Execute the notebook**: Run all cells in the provided Jupyter/Colab notebook sequentially.
```
