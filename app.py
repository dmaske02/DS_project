# ============================================================
#     FINAL — CLEAN, ERROR-FREE STREAMLIT ANALYTICS APP
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# ============================================================
# UI CONFIG (Dark Theme + Wide Layout)
# ============================================================
st.set_page_config(
    page_title="Census Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stApp {background-color: #0E1117;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Upload Data", "EDA", "Outlier Detection", "ML Model"]
)

# ============================================================
# PAGE 1 — UPLOAD
# ============================================================
if page == "Upload Data":
    st.title("📥 Upload Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file.")

if "df" not in st.session_state:
    if page != "Upload Data":
        st.error("Please upload a dataset first.")
        st.stop()

df = st.session_state["df"].copy()

# ============================================================
# AUTO FIX — Convert numeric-text to numeric
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

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("❗ Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("📊 Summary Statistics")
    if numerics:
        st.write(df[numerics].describe())
    else:
        st.warning("No numeric columns found.")

    if numerics:
        st.subheader("📈 Distribution Plot")
        col = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    if categoricals:
        st.subheader("📊 Categorical Plot")
        cat = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[cat].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    if len(numerics) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numerics].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)
    else:
        st.warning("Need minimum 2 numeric columns for heatmap.")

# ============================================================
# PAGE 3 — OUTLIER DETECTION
# ============================================================
if page == "Outlier Detection":
    st.title("🚨 Outlier Detection (IQR Method)")

    if not numerics:
        st.warning("No numeric columns available.")
        st.stop()

    col = st.selectbox("Select column", numerics)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.write(f"Lower Bound: {lower}")
    st.write(f"Upper Bound: {upper}")

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"Total Outliers Found: {outliers.shape[0]}")
    st.dataframe(outliers.head())

    st.subheader("📦 Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

# ============================================================
# PAGE 4 — ML MODEL
# ============================================================
if page == "ML Model":
    st.title("🤖 Machine Learning Model")

    target = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    model_type = st.radio("Choose model type", ["Regression", "Classification"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
    except:
        st.error("Your target column cannot be used for ML.")
        st.stop()

    # ---------------------- REGRESSION ----------------------
    if model_type == "Regression":
        if y.dtype == "object":
            st.error("Target is categorical. Choose classification.")
            st.stop()

        model = RandomForestRegressor()

        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        st.subheader("📈 Regression Results")
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("RMSE:", mean_squared_error(y_test, preds)**0.5)

    # ---------------------- CLASSIFICATION ----------------------
    else:
        if y.dtype != "object":
            st.error("Target is numeric. Choose regression.")
            st.stop()

        model = RandomForestClassifier()

        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        st.subheader("📈 Classification Results")
        st.write("Accuracy:", accuracy_score(y_test, preds))

        st.success("Model trained successfully!")
