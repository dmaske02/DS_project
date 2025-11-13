# ============================================================
#       FINAL CLEAN STREAMLIT APP (NO DARK MODE + NO ERRORS)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Census Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Upload Data", "EDA", "Outlier Detection", "ML Model"]
)

# ============================================================
# PAGE 1 — UPLOAD DATA
# ============================================================
if page == "Upload Data":
    st.title("📥 Upload Your Dataset")

    file = st.file_uploader("Upload CSV file", type=['csv'])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to continue.")

# ============================================================
# SAFE LOAD — PROTECT AGAINST KEYERROR
# ============================================================
if "df" not in st.session_state:
    if page != "Upload Data":
        st.error("⚠ Please upload a dataset first from the sidebar.")
        st.stop()

df = st.session_state.df.copy()

# ============================================================
# AUTO FIX — Convert numeric text to numbers
# ============================================================
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", "").str.strip()
    df[col] = pd.to_numeric(df[col], errors="ignore")

numerics = df.select_dtypes(include=['int64','float64']).columns.tolist()
categoricals = df.select_dtypes(include=['object','category','bool']).columns.tolist()

# ============================================================
# PAGE 2 — EDA
# ============================================================
if page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Missing
    st.subheader("❗ Missing Values")
    st.dataframe(df.isnull().sum())

    # Summary
    st.subheader("📊 Summary Statistics (Numeric)")
    if numerics:
        st.write(df[numerics].describe())
    else:
        st.warning("No numeric columns found.")

    # Distribution
    if numerics:
        st.subheader("📈 Distribution Plot")
        col = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Categorical plot
    if categoricals:
        st.subheader("📊 Categorical Plot")
        cat = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[cat].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    if len(numerics) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numerics].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for heatmap.")

# ============================================================
# PAGE 3 — OUTLIER DETECTION
# ============================================================
if page == "Outlier Detection":
    st.title("🚨 Outlier Detection (IQR Method)")

    if not numerics:
        st.warning("Dataset has no numeric columns.")
        st.stop()

    col = st.selectbox("Select a numeric column", numerics)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.write(f"Lower Bound: {lower}")
    st.write(f"Upper Bound: {upper}")

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"Total Outliers Found: **{outliers.shape[0]}**")

    st.dataframe(outliers.head())

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

# ============================================================
# PAGE 4 — MACHINE LEARNING MODEL
# ============================================================
if page == "ML Model":
    st.title("🤖 Machine Learning Model")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['float64','int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    model_type = st.radio("Select model type", ["Regression", "Classification"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
    except:
        st.error("Your target column cannot be used for ML.")
        st.stop()

    # ---------------------- Regression ------------------------
    if model_type == "Regression":
        if y.dtype == "object":
            st.error("Target is categorical. Choose Classification.")
            st.stop()

        model = RandomForestRegressor()

        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("📈 Regression Results")
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("RMSE:", mean_squared_error(y_test, preds)**0.5)

    # ---------------------- Classification ------------------------
    else:
        if y.dtype != "object":
            st.error("Target is numeric. Choose Regression.")
            st.stop()

        model = RandomForestClassifier()

        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("📈 Classification Results")
        st.write("Accuracy:", accuracy_score(y_test, preds))

        st.success("Model training completed successfully!")
