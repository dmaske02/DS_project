# ==============================================================
#            FINAL STABLE STREAMLIT APP (HEATMAP FIXED)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# --------------------------------------------------------------
# APP CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Census Dashboard", layout="wide")

# --------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To Page",
    ["Upload Data", "EDA", "Outlier Detection", "ML Model"]
)

# --------------------------------------------------------------
# PAGE 1 — UPLOAD
# --------------------------------------------------------------
if page == "Upload Data":
    st.title("📥 Upload Your Dataset")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            st.session_state["df"] = df
            st.success("Dataset uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("Please upload a CSV file to continue.")

# --------------------------------------------------------------
# LOAD DF SAFELY
# --------------------------------------------------------------
df = st.session_state.get("df", None)

if df is None and page != "Upload Data":
    st.error("⚠ Please upload a dataset first.")
    st.stop()

if df is not None:
    df = df.copy()

# --------------------------------------------------------------
# ULTRA-ROBUST NUMERIC DETECTION (works for ANY dataset)
# --------------------------------------------------------------

def convert_to_numeric(series):
    # Remove everything except digits, dot, minus
    cleaned = series.astype(str).apply(lambda x: re.sub(r"[^0-9.\-]", "", x))
    return pd.to_numeric(cleaned, errors="coerce")

numeric_df = pd.DataFrame()
numerics = []
categoricals = []

for col in df.columns:
    try:
        converted = convert_to_numeric(df[col])

        # retain if numeric column has at least 5 valid values
        if converted.count() >= 5:
            numeric_df[col] = converted
            numerics.append(col)
        else:
            categoricals.append(col)
    except:
        categoricals.append(col)

# --------------------------------------------------------------
# PAGE 2 — EDA
# --------------------------------------------------------------
if page == "EDA" and df is not None:
    st.title("📊 Exploratory Data Analysis")

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # Missing Values
    st.subheader("❗ Missing Values")

    try:
        mv = df.isna().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        st.dataframe(mv)
    except Exception as e:
        st.error(f"Missing values error: {e}")

    # Summary
    st.subheader("📊 Summary Statistics")

    if numerics:
        st.dataframe(numeric_df[numerics].describe().T)
    else:
        st.warning("No numeric columns detected.")

    # Distribution
    if numerics:
        st.subheader("📈 Distribution Plot")
        col = st.selectbox("Select Numeric Column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(numeric_df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Categorical plot
    if categoricals:
        st.subheader("📊 Categorical Value Counts")
        col = st.selectbox("Select Categorical Column", categoricals)
        fig, ax = plt.subplots()
        df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap (Guaranteed Working)")

    if len(numerics) < 2:
        st.warning("Not enough numeric columns for a correlation heatmap.")
    else:
        try:
            corr = numeric_df[numerics].corr()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Heatmap error: {e}")

# --------------------------------------------------------------
# PAGE 3 — OUTLIER DETECTION
# --------------------------------------------------------------
if page == "Outlier Detection" and df is not None:
    st.title("🚨 Outlier Detection (IQR Method)")

    if not numerics:
        st.warning("No numeric columns available.")
        st.stop()

    col = st.selectbox("Select Numeric Column", numerics)

    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.write(f"Lower Bound = {lower}")
    st.write(f"Upper Bound = {upper}")

    outliers = df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
    st.write(f"Outliers Found: **{outliers.shape[0]}**")
    st.dataframe(outliers.head())

    fig, ax = plt.subplots()
    sns.boxplot(x=numeric_df[col], ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------
# PAGE 4 — MACHINE LEARNING
# --------------------------------------------------------------
if page == "ML Model" and df is not None:
    st.title("🤖 Machine Learning Model")

    target = st.selectbox("Select Target Column", df.columns)

    X = numeric_df.copy()
    if target in X.columns:
        X = X.drop(columns=[target])

    y = df[target]

    num_cols = [c for c in numerics if c != target]
    cat_cols = [c for c in categoricals if c in df.columns]

    # Preprocessor
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_cols
        ))

    preprocessor = ColumnTransformer(transformers)

    model_type = st.radio("Select Model Type", ["Regression", "Classification"])

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            pd.concat([numeric_df, df[cat_cols]], axis=1),
            y,
            test_size=0.2
        )
    except Exception as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Regression
    if model_type == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target must be numeric for regression.")
            st.stop()

        model = RandomForestRegressor()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("📈 Regression Results")
        st.write("R² Score:", round(r2_score(y_test, preds), 4))
        st.write("RMSE:", round(mean_squared_error(y_test, preds)**0.5, 4))

    # Classification
    else:
        if pd.api.types.is_numeric_dtype(y):
            st.error("Target must be categorical for classification.")
            st.stop()

        model = RandomForestClassifier()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("📈 Classification Results")
        st.write("Accuracy:", round(accuracy_score(y_test, preds), 4))

        st.success("Model training completed!")
