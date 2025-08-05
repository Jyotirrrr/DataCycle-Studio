import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

st.title("📊 Interactive Data Analysis & ML App - End-to-End Project with Dashboard")

file = st.file_uploader("📁 Upload CSV or XLSX", type=["csv", "xlsx"])
df = None

def plot_interactive_dashboard(df):
    st.subheader("📊 Interactive PowerBI/Tableau-style Dashboard")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not numeric_cols and not cat_cols:
        st.info("No suitable columns for dashboard.")
        return

    chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Box", "Histogram", "Pie"], key="chart_type")
    x_axis = st.selectbox("X-axis", df.columns, key="x_axis_dashboard")
    y_axis = None
    if chart_type in ["Scatter", "Bar", "Box", "Histogram"]:
        y_options = [None] + list(df.columns)
        y_axis = st.selectbox("Y-axis", y_options, key="y_axis_dashboard")
        if not y_axis or y_axis == x_axis:
            y_axis = None
    color_by = st.selectbox("Color By", [None] + cat_cols, key="color_by_dashboard")

    fig = None
    if chart_type == "Scatter" and y_axis:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Bar" and y_axis:
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Box" and y_axis:
        fig = px.box(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, color=color_by)
    elif chart_type == "Pie":
        fig = px.pie(df, names=x_axis, color=color_by)

    if fig:
        st.plotly_chart(fig, use_container_width=True)

if file:
    # Load Data
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    st.write("## 🧾 Data Preview")
    preview = st.radio("Show", ["Head", "Tail"])
    st.dataframe(df.head() if preview == "Head" else df.tail())

    # Data Info and Obvious Type Conflicts
    st.write("## 📌 Data Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Data Types:")
    st.dataframe(df.dtypes.astype(str))
    with st.expander("🔎 Detect Type Mismatches (e.g., numbers stored as text)"):
        suspected_numeric = []
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col].astype(float)
                    suspected_numeric.append(col)
                except:
                    pass
        if suspected_numeric:
            st.warning(f"Columns suspected as numeric but stored as text: {suspected_numeric}")
        else:
            st.success("No obvious data type mismatches detected.")

    # Nulls and Duplicates
    st.write("## 🚨 Null & Duplicate Values")
    st.write("Null values per column:")
    st.dataframe(df.isnull().sum())
    st.write(f"Duplicate rows: {df.duplicated().sum()}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧼 Fill Nulls with Mean (Numerical)"):
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("Filled missing values with column means.")
            st.write("Null values after fill:")
            st.dataframe(df.isnull().sum())
    with col2:
        if st.button("🗑️ Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicate rows removed.")
            st.write(f"Remaining rows: {df.shape[0]}")

    # Categorical Analysis
    st.write("## 🏷️ Categorical Columns Overview")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        st.write(f"**{col}:** {df[col].nunique()} unique values")
        st.write("Distribution:")
        st.dataframe(df[col].value_counts())

    # Consistency Checks
    st.write("##  Consistency & Data Range Checks")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if (df[col] < 0).any():
            st.warning(f"Column '{col}' has negative values. Please check if this is expected (e.g., Age, Salary).")
    with st.expander("Check Related Columns Consistency (e.g., Start-End Dates):"):
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if len(date_cols) >= 2:
            st.write("If you have start/end dates, check below:")
            st.dataframe(df[date_cols])
            try:
                inconsistent = (pd.to_datetime(df[date_cols[0]]) > pd.to_datetime(df[date_cols[1]])).sum()
                st.info(f"{inconsistent} records where {date_cols[0]} is after {date_cols[1]}")
            except:
                st.warning("Date consistency could not be checked (please ensure date columns are properly formatted).")
        else:
            st.write("No obvious start/end date columns detected.")

    # Outlier Detection
    st.write("## 🚦 Outlier Detection")
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            st.warning(f"{outliers} outliers detected in '{col}' (IQR method).")
        else:
            st.success(f"No significant outliers detected in '{col}'.")

    # Visualizations
    st.write("## 📊 Visualization")
    vis_tab = st.selectbox("Choose Visualization Type", ["Histogram", "Boxplot", "Pie Chart (Categorical)", "Value Counts (Categorical)"])
    if vis_tab == "Histogram":
        col = st.selectbox("Select Numerical Column", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, color="skyblue")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    elif vis_tab == "Boxplot":
        col = st.selectbox("Select Numerical Column", num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(df[col], ax=ax, color="lightgray")
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
    elif vis_tab == "Pie Chart (Categorical)":
        col = st.selectbox("Select Categorical Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    elif vis_tab == "Value Counts (Categorical)":
        col = st.selectbox("Select Categorical Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot.bar(color="purple", ax=ax)
        ax.set_title(f"Value Counts for {col}")
        st.pyplot(fig)

    # Data Summary
    st.write("## 📈 Data Summary")
    st.write("### Numeric Summary (mean, median, std, min, max, percentiles)")
    summary = df.describe().T
    summary["median"] = df.median(numeric_only=True)
    summary["mode"] = df.mode(numeric_only=True).iloc[0]
    st.dataframe(summary)
    st.download_button(
        label="⬇️ Download Cleaned Data",
        data=df.to_csv(index=False).encode(),
        file_name="cleaned_data.csv"
    )

    st.write("---")
    st.write("##  Still wondering what would the accuracy be??")
    st.markdown(
        """
This section will walk through:
- Data Preprocessing: Feature selection, encoding, scaling, addressing outliers, train-test split, class balancing
- Model Training: Model choice, cross-validation, regularization, metrics
- Model Evaluation: Test scores, interpretation, reports.
        """
    )

    # --- Data Preprocessing Section ---
    st.subheader("🧪 Data Preprocessing Steps")
    # --- Feature Selection ---
    with st.expander("1️⃣ Feature Selection & Data Preparation", expanded=True):
        st.write("Detecting and selecting the most relevant features for modeling...")
        target = st.selectbox("🎯 Select Your Target Column", df.columns)
        features = [c for c in df.columns if c != target]
        selected_features = st.multiselect("👉 Select Feature Columns (Auto: All except Target)", features, default=features)
        st.info(f"Selected Features: {selected_features}")

        X = df[selected_features]
        y = df[target]

        # Handle missing values for numeric and categorical
        for c in X.columns:
            if X[c].isnull().any():
                if X[c].dtype in [np.float64, np.int64]:
                    X[c] = X[c].fillna(X[c].mean())
                else:
                    X[c] = X[c].fillna(X[c].mode()[0])
        # Encoding categorical features
        for c in X.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
        # Detect classification vs regression
        problem_type = "classification" if (y.dtype == 'object' or y.nunique() < 10) else "regression"
        st.info(f"Problem Type Detected: **{problem_type.title()}**")

        # Encode target if classification
        if problem_type == "classification" and y.dtype in ['object', 'category']:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

    # --- Outlier Handling and Scaling ---
    with st.expander("2️⃣ Outlier Handling & Feature Scaling", expanded=True):
        st.write("Addressing outliers and scaling features...")
        for col in X.select_dtypes(include=[np.number]).columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((X[col] < lower) | (X[col] > upper)).sum()
            if outliers > 0:
                st.warning(f"{outliers} outliers capped in column {col}.")
                X[col] = np.where(X[col] < lower, lower, np.where(X[col] > upper, upper, X[col]))
        scaler = st.selectbox("Choose Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
        if scaler == "StandardScaler":
            X[X.select_dtypes(include=[np.number]).columns] = StandardScaler().fit_transform(X.select_dtypes(include=[np.number]))
        elif scaler == "MinMaxScaler":
            X[X.select_dtypes(include=[np.number]).columns] = MinMaxScaler().fit_transform(X.select_dtypes(include=[np.number]))
        st.success("Scaling and outlier treatment done.")

    # --- Train-Test Split and Class Balancing ---
    with st.expander("3️⃣ Train-Test Split & Class Balancing", expanded=True):
        st.write("Splitting data and handling any class imbalance if needed...")
        if problem_type == "classification":
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() < 2:
                st.warning("Some classes have less than 2 samples. Stratified splitting is not possible. Using regular split instead.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            # Apply SMOTE if needed
            if len(np.unique(y_train)) == 2 and (pd.Series(y_train).value_counts().min() / pd.Series(y_train).value_counts().max()) < 0.4:
                st.warning("Class imbalance detected! Applying SMOTE for balancing...")
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("Train-test split complete.")

    # --- Model Training Section ---
    st.subheader("⚙️ Model Training & Validation")
    model_choice = st.radio("Choose Model", ["Random Forest", "Linear/XGB", "Logistic Regression (Classification Only)"])
    with st.expander("4️⃣ Training and Cross-Validation", expanded=True):
        st.write("Using cross-validation and grid search for hyperparameter tuning...")

        if model_choice == "Random Forest":
            model = RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor()
            param_grid = {"n_estimators": [50, 100]}
        elif model_choice == "Linear/XGB":
            model = xgb.XGBClassifier(eval_metric='mlogloss') if problem_type == "classification" else xgb.XGBRegressor()
            param_grid = {"max_depth": [3, 5], "n_estimators": [50, 100]}
        elif model_choice == "Logistic Regression (Classification Only)":
            if problem_type == "classification":
                model = LogisticRegression()
                param_grid = {"C": [0.1, 1, 10]}
            else:
                st.error("Logistic Regression is only for classification.")
                st.stop()

        try:
            grid = GridSearchCV(model, param_grid, cv=3)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            st.info(f"Best Parameters: {grid.best_params_}")
        except Exception as e:
            st.warning(f"Grid search failed: {e}")
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if problem_type == "regression":
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc:.3f}")
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred, zero_division=0))

    # SHAP Analysis - at root level, not nested
    with st.expander("5️⃣ Model Interpretation (SHAP)", expanded=True):
        st.write("Interpreting the model with SHAP values:")
        try:
            if problem_type == "regression":
                explainer = shap.Explainer(model, X_train, feature_names=X.columns)
                shap_values = explainer(X_test)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
        except Exception as e:
            st.warning(f"SHAP plot could not be generated: {e}")

    # Post-training evaluation and report
    st.subheader("📋 Post-Training: Detailed Evaluation & Report")
    with st.expander("6️⃣ After Training - Evaluation, Error Analysis, Ensembling, Monitoring"):
        st.markdown(
            """**Actions Taken:**
- Model evaluated on test set using standard metrics.
- SHAP for interpretation.
- For further accuracy improvements, consider ensembling (bagging, boosting), model calibration, and regular model monitoring.
---
**Next Steps for Real Deployment:**
- Wrap preprocessing and modeling in a pipeline.
- Monitor predictions and retrain on new data as it arrives for drift."""
        )

    st.write("### 🏁 Summary of All Steps")
    st.info(
        """- Data checked for nulls, duplicates, type mismatches, ranges, and categorical structure.
- Outliers capped and features scaled (as selected).
- Categorical features encoded.
- Data split for validation in a robust (stratified if classification) way.
- Class imbalance handled if needed.
- Model selection included cross-validation and grid search for best params.
- Model interpretation provided with SHAP.
- All findings, like distributions and outliers, visualized and downloadable."""
    )

    # End-to-End Interactive Dashboard after Model Training
    plot_interactive_dashboard(df)

st.sidebar.success("Project by Jyotiraditya Jadhav. 🚀 Data Analyst & ML (End-to-End + BI Dashboard)")
