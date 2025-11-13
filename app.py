# final_app.py — Robust Streamlit EDA + Outlier + ML (no dark mode)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Census Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "EDA", "Outlier Detection", "ML Model"])

# ---------------------------
# Upload page
# ---------------------------
if page == "Upload Data":
    st.title("Upload your CSV dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # store in session state (dict-style)
        st.session_state["df"] = df
        st.success("File uploaded and stored in session. Go to other pages via the sidebar.")
        st.subheader("Preview")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to proceed.")

# ---------------------------
# Ensure df exists in session_state for the other pages
# ---------------------------
df = st.session_state.get("df", None)
if df is None and page != "Upload Data":
    st.error("No dataset loaded. Please upload a CSV first on the 'Upload Data' page.")
    st.stop()

# Work on a local copy
df = df.copy()

# ---------------------------
# AUTO CONVERT numeric-looking columns
# ---------------------------
for col in df.columns:
    # remove commas and whitespace then attempt conversion
    try:
        # `.str` methods require strings — convert temporarily
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except Exception:
        # if column is not string-like or conversion fails, leave as-is
        pass

numerics = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categoricals = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# ---------------------------
# EDA page
# ---------------------------
if page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("Missing values (per column)")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0].sort_values(ascending=False))

    st.subheader("Basic summary (numeric)")
    if numerics:
        st.dataframe(df[numerics].describe().T)
    else:
        st.warning("No numeric columns detected after conversion.")

    if numerics:
        st.subheader("Distribution plot")
        sel_num = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[sel_num].dropna(), kde=True, ax=ax)
        ax.set_xlabel(sel_num)
        st.pyplot(fig)

    if categoricals:
        st.subheader("Categorical counts")
        sel_cat = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[sel_cat].value_counts(dropna=False).plot(kind="bar", ax=ax)
        ax.set_ylabel("count")
        st.pyplot(fig)

    st.subheader("Correlation heatmap (numeric columns)")
    if len(numerics) >= 2:
        corr = df[numerics].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns to show correlation heatmap.")

# ---------------------------
# Outlier Detection page
# ---------------------------
if page == "Outlier Detection":
    st.title("Outlier Detection (IQR method)")

    if not numerics:
        st.warning("No numeric columns detected — outlier detection requires numeric data.")
    else:
        col = st.selectbox("Choose numeric column", numerics)
        # compute IQR bounds
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        st.write(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
        st.write(f"Lower bound = {lower}, Upper bound = {upper}")

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        st.write(f"Number of outliers detected: {len(outliers)}")
        st.dataframe(outliers.head(50))

        st.subheader("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax)
        st.pyplot(fig)

        st.subheader("Options")
        remove_option = st.checkbox("Remove these outliers from the dataset (creates cleaned copy)")
        if remove_option:
            df_clean = df[~((df[col] < lower) | (df[col] > upper))].reset_index(drop=True)
            st.session_state["df_clean"] = df_clean
            st.success(f"Outliers removed. Cleaned dataset stored as `df_clean` in session (rows: {len(df_clean)})")
            st.dataframe(df_clean.head())

# ---------------------------
# ML Model page
# ---------------------------
if page == "ML Model":
    st.title("Machine Learning (Regression & Classification)")

    st.write("Select a target column and model type. The app builds a simple RandomForest pipeline with basic imputation and one-hot encoding.")

    target = st.selectbox("Select target column", df.columns)

    if target is None:
        st.error("Please select a target column.")
        st.stop()

    # Build feature & label sets
    X = df.drop(columns=[target])
    y = df[target]

    # detect column types for preprocessor
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    st.write(f"Detected numeric features: {num_cols}")
    st.write(f"Detected categorical features: {cat_cols}")

    # safe preprocessor: if no columns in a group, skip that transformer
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        # OneHotEncoder with sparse_output for sklearn >=1.4 compatibility
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # fallback for older sklearn versions
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols))

    if not transformers:
        st.error("No usable feature columns detected. ML requires at least one feature column.")
        st.stop()

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model_type = st.radio("Choose task type", ("Regression", "Classification"))

    # train/test split (wrap in try for safety)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Failed to split data. Check your target and features. Error: {e}")
        st.stop()

    # build pipeline & fit
    if model_type == "Regression":
        # ensure target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Selected target is not numeric — choose Classification.")
            st.stop()

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])

        with st.spinner("Training regression model..."):
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = mean_squared_error(y_test, preds, squared=False)
                st.subheader("Regression Results")
                st.write(f"R²: {r2:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                # store model in session for later use
                st.session_state["trained_model"] = pipeline
            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        # Classification
        if pd.api.types.is_numeric_dtype(y):
            st.error("Selected target is numeric — choose Regression.")
            st.stop()

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        pipeline = Pipeline([("pre", preprocessor), ("model", model)])

        with st.spinner("Training classification model..."):
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.subheader("Classification Results")
                st.write(f"Accuracy: {acc:.4f}")
                st.session_state["trained_model"] = pipeline
            except Exception as e:
                st.error(f"Training failed: {e}")

    # optional: save model
    if "trained_model" in st.session_state:
        if st.button("Save trained model to file"):
            import joblib, datetime
            fname = f"trained_model_{int(datetime.datetime.now().timestamp())}.joblib"
            try:
                joblib.dump(st.session_state["trained_model"], fname)
                st.success(f"Model saved to `{fname}`")
            except Exception as e:
                st.error(f"Failed to save model: {e}")

# ---------------------------
# End of app
# ---------------------------
