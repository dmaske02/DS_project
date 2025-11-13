# ==============================================================
#                 FINAL ERROR-FREE STREAMLIT APP
# ==============================================================

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

# --------------------------------------------------------------
# App Config
# --------------------------------------------------------------
st.set_page_config(page_title="Census Dashboard", layout="wide")

# --------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To Page",
    ["Upload Data", "EDA", "Outlier Detection", "ML Model"]
)

# --------------------------------------------------------------
# Page 1 — Upload Page
# --------------------------------------------------------------
if page == "Upload Data":
    st.title("📥 Upload Your Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            st.session_state["df"] = df
            st.success("Dataset uploaded!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
    else:
        st.info("Upload a CSV file to proceed.")

# --------------------------------------------------------------
# Safe load df for all other pages
# --------------------------------------------------------------
df = st.session_state.get("df", None)

if df is None and page != "Upload Data":
    st.error("⚠ Please upload a dataset first from the 'Upload Data' page.")
    st.stop()

# Work on safe copy
if df is not None:
    df = df.copy()

# --------------------------------------------------------------
# AUTO CLEAN — Convert numeric-looking values
# --------------------------------------------------------------
if df is not None:
    for col in df.columns:
        try:
            df[col] = df[col].astype(str).str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except:
            pass

    numerics = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categoricals = df.select_dtypes(include=['object','category','bool']).columns.tolist()

# --------------------------------------------------------------
# Page 2 — EDA
# --------------------------------------------------------------
if page == "EDA" and df is not None:
    st.title("📊 Exploratory Data Analysis")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # --------------------------------------
    # Missing Value Section (Safe)
    # --------------------------------------
    st.subheader("❗ Missing Values")

    try:
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        st.dataframe(mv)
    except Exception as e:
        st.error(f"Unable to compute missing values: {e}")

    # --------------------------------------
    # Summary Statistics
    # --------------------------------------
    st.subheader("📊 Summary Statistics")

    if numerics:
        st.dataframe(df[numerics].describe().T)
    else:
        st.warning("No numeric columns detected.")

    # --------------------------------------
    # Distribution Plot
    # --------------------------------------
    if numerics:
        st.subheader("📈 Distribution Plot")
        col = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # --------------------------------------
    # Categorical Count Plot
    # --------------------------------------
    if categoricals:
        st.subheader("📊 Categorical Value Counts")
        col = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # --------------------------------------
    # SAFE HEATMAP (Auto-Clean)
    # --------------------------------------
    st.subheader("🔥 Correlation Heatmap (Auto Cleaned)")

    numeric_df = pd.DataFrame()

    # Build numeric_df safely
    for col in df.columns:
        try:
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.count() > 2:
                numeric_df[col] = temp
        except:
            pass

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns for a heatmap.")
    else:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Heatmap error: {e}")

# --------------------------------------------------------------
# Page 3 — Outlier Detection
# --------------------------------------------------------------
if page == "Outlier Detection" and df is not None:
    st.title("🚨 Outlier Detection (IQR Method)")

    if not numerics:
        st.warning("No numeric columns available.")
        st.stop()

    col = st.selectbox("Select Numeric Column", numerics)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.write(f"Lower Bound = {lower}")
    st.write(f"Upper Bound = {upper}")

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"Outliers Found: {outliers.shape[0]}")
    st.dataframe(outliers.head())

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------
# Page 4 — Machine Learning
# --------------------------------------------------------------
if page == "ML Model" and df is not None:
    st.title("🤖 Machine Learning Model")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # detect column types
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Safe Preprocessor
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe)
            ]),
            cat_cols
        ))

    preprocessor = ColumnTransformer(transformers)

    model_type = st.radio("Select Model Type", ["Regression", "Classification"])

    # Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except Exception as e:
        st.error(f"Train-test split error: {e}")
        st.stop()

    # -----------------------
    # REGRESSION
    # -----------------------
    if model_type == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target must be numeric for Regression.")
            st.stop()

        model = RandomForestRegressor()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("Regression Results")
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("RMSE:", mean_squared_error(y_test, preds)**0.5)

    # -----------------------
    # CLASSIFICATION
    # -----------------------
    else:
        if pd.api.types.is_numeric_dtype(y):
            st.error("Target must be categorical for Classification.")
            st.stop()

        model = RandomForestClassifier()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("Classification Results")
        st.write("Accuracy:", accuracy_score(y_test, preds))

        st.success("Model trained successfully!")
