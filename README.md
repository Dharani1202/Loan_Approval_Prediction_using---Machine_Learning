# Loan Approval Prediction Using Machine Learning

This project predicts loan approval status using machine learning techniques. The goal is to analyze the dataset, handle data issues, and train a predictive model to classify loan approvals effectively.

* <a href="https://github.com/Dharani1202/Loan_Approval_Prediction_using---Machine_Learning/blob/main/Insurance%20Claim%20Fraud%20Detection.ipynb"> View the Project </a>

## About

The project uses a dataset containing loan applicant information such as income, credit history, education, and other features to predict whether a loan will be approved. Data preprocessing, feature engineering, scaling, balancing, and machine learning modeling were performed step by step.

All code and analysis are documented in a Jupyter Notebook for easy understanding.

## Tools & Technologies Used

* **Python (3.x)**
* **Pandas** – for data cleaning and manipulation
* **NumPy** – for numerical operations
* **Matplotlib / Seaborn** – for visualization
* **scikit-learn (sklearn)** – for preprocessing, modeling, and evaluation
* **SMOTE (from imbalanced-learn)** – to balance the dataset
* **Jupyter Notebook** – for interactive analysis

## Data Cleaning & Preprocessing

1. **Handle Missing Values and Duplicates**
   Checked for null values using `isnull()` and removed duplicates for data accuracy.

2. **Dataset Overview**
   Examined `shape`, `info()`, `columns`, and `describe()` for statistical analysis and data understanding.

3. **Outlier Detection and Removal**

   * Used **Z-score** and **box plots** to identify outliers in numerical columns.
   * Removed outliers to improve model performance.

4. **Scaling & Skewness Handling**

   * Applied scaling to reduce bias in input variables.
   * Used **StandardScaler** for standardization of numerical features.
   * Checked skewness and adjusted distributions where necessary.

5. **Feature Engineering**

   * Separated numerical columns into categorical columns where appropriate.
   * Performed **Label Encoding** on categorical variables for model compatibility.

6. **Balancing Dataset**
   Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset for better classification accuracy.

7. **Train-Test Split**
   Split the dataset into training and testing sets using `train_test_split`.

## Modeling & Evaluation

1. **Model Used**

   * **Logistic Regression** from scikit-learn.

2. **Training the Model**
   Used training data to fit the Logistic Regression model.

3. **Evaluation Metrics**

   * Accuracy Score (`accuracy_score`)
   * Confusion Matrix (`confusion_matrix`)
   * Classification Report (`classification_report`)
   * ROC Curve (`roc_curve`)

4. **Cross-Validation**
   Applied **K-Fold Cross Validation** to validate the model’s generalization using `cross_val_score`.

## Key Insights

* Proper preprocessing (handling missing values, removing outliers, scaling, and balancing) significantly improved model accuracy.
* Logistic Regression was able to classify loan approval effectively with good performance metrics.
* Balanced dataset via SMOTE helped mitigate bias towards the majority class.

## Conclusion

This project demonstrates the end-to-end workflow of a machine learning classification problem: from cleaning and preprocessing data to training, evaluating, and validating a predictive model for loan approval.


* <a href="https://github.com/Dharani1202/Loan_Approval_Prediction_using---Machine_Learning/blob/main/Insurance%20Claim%20Fraud%20Detection.ipynb"> View the Project </a>
