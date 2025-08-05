# DataCycle Studio <br/>
Check out this app by click on this <a href="imbalanced-learn"> Click here</a> <br/>
## Overview <br/>
This project is a comprehensive, interactive web application built with Streamlit that provides an end-to-end platform for data analysis and machine learning. The app guides users through a full data science workflow, from raw data ingestion and exploration to advanced predictive modeling and model interpretability.

Designed as a no-code tool, this application makes powerful data analysis accessible to users of all technical backgrounds while serving as a robust portfolio piece demonstrating a complete understanding of a data science pipeline.

# Key Features
## 📂 Data Ingestion & Exploration
Multi-Format Support: Easily upload datasets in both CSV and XLSX formats.

Data Preview: Instantly view the head and tail of the dataset.

Data Integrity Checks: Automatically report on the number of rows, columns, data types, and count of null and duplicate values.

Type Mismatch Detection: Intelligent checks to identify columns that are numeric in nature but stored as text.

## 🧹 Data Cleaning & Preprocessing
Null Value Handling: One-click option to fill all missing numerical values using the mean of their respective columns.

Duplicate Removal: Easily identify and remove duplicate rows to ensure data quality.

Outlier & Consistency Checks: Automated warnings for negative values in columns like Age or Salary, and IQR method-based outlier detection for all numerical features.

## 📊 Interactive Visualizations
Dynamic Dashboard: Generate interactive charts (Scatter, Bar, Box, Histogram, Pie) in a PowerBI/Tableau-style dashboard using Plotly.

Statistical Plots: Create histograms and box plots to analyze data distributions and visualize outliers.

## 🤖 Predictive Modeling & Advanced Techniques
This section is the core of the app and demonstrates a deep understanding of the machine learning lifecycle.

Problem Type Detection: The app automatically determines if the problem is Classification or Regression based on the target column's data.

Advanced Preprocessing: The workflow automatically performs feature selection, categorical encoding, outlier capping, and feature scaling (StandardScaler, MinMaxScaler) before training.

Class Balancing (SMOTE): For imbalanced classification datasets, the app automatically applies the SMOTE technique to balance the classes, ensuring more reliable model training.

Model Selection & Tuning: Choose from powerful models like RandomForest, XGBoost, and Logistic Regression. The app uses GridSearchCV to find the best hyperparameters, demonstrating a mastery of model tuning.

Performance Metrics: For both regression and classification, the app provides a full suite of metrics, including R², RMSE, Accuracy, and a full Classification Report.

Model Interpretation (SHAP): Gain deep insights into model behavior with integrated SHAP (SHapley Additive exPlanations) plots, which show how each feature contributes to the final prediction.

## ⬇️ Export
Download Cleaned Data: Download the processed and cleaned dataset as a CSV file for use in other projects or tools.

## Technology Stack
Framework: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn

Visualization: Plotly, Seaborn, Matplotlib

Model Interpretation: SHAP

## How to Run the App
1) Clone the Repository:
```bash
git clone https://github.com/your-username/DataCycle-Studio.git
cd DataCycle-Studio
```


2) Create a Virtual Environment: (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3) Install Dependencies:
```bash
pip install -r requirements.txt
```

4) Run the Application:
```bash
streamlit run app.py
```

5) The app will open in your default web browser at <a href="http://localhost:8501">http://localhost:8501.</a>
