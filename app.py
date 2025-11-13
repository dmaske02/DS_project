# ==============================================================
#        FINAL STREAMLIT APP (NO HEATMAP, NO COPY ERROR)
# ==============================================================

import streamlit as st
import pandas as pd
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
# PAGE 1 — UPLOAD DATA
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
            st.error(f"Error reading CSV: {e}")

    else:
        st.info("Upload a CSV file to continue.")

# --------------------------------------------------------------
# SAFE LOAD DF
# --------------------------------------------------------------
if "df" in st.session_state:
    df = st.session_state["df"]      # SAFE → df exists
else:
    df = None                        # df does NOT exist

# Protect other pages
if page != "Upload Data" and df is None:
    st.error("⚠ Please upload a dataset first.")
    st.stop()

# --------------------------------------------------------------
# PAGE 2 — EDA (NO HEATMAP)
# --------------------------------------------------------------
if page == "EDA" and df is not None:
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Missing values
    st.subheader("❗ Missing Values")
    mv = df.isna().sum().reset_index()
    mv.columns = ["Column", "Missing Count"]
    st.dataframe(mv)

    # Numeric and categorical detection
    numerics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricals = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    # Summary stats
    st.subheader("📊 Summary Statistics")
    if numerics:
        st.dataframe(df[numerics].describe().T)
    else:
        st.warning("No numeric columns found.")

    # Distribution plot
    if numerics:
        st.subheader("📈 Distribution Plot")
        col = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Categorical bar plot
    if categoricals:
        st.subheader("📊 Categorical Value Counts")
        col = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
        st.pyplot(fig)

# --------------------------------------------------------------
# PAGE 3 — OUTLIER DETECTION
# --------------------------------------------------------------
if page == "Outlier Detection" and df is not None:
    st.title("🚨 Outlier Detection (IQR Method)")

    numerics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numerics:
        st.warning("No numeric columns found.")
        st.stop()

    col = st.selectbox("Select numeric column", numerics)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    st.write(f"Lower Bound: {lower}")
    st.write(f"Upper Bound: {upper}")

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"Outliers Found: {outliers.shape[0]}")
    st.dataframe(outliers.head())

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------
# PAGE 4 — MACHINE LEARNING
# --------------------------------------------------------------
if page == "ML Model" and df is not None:
    st.title("🤖 Machine Learning Model")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','bool','category']).columns.tolist()

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

    model_type = st.radio("Model Type", ["Regression", "Classification"])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
    except Exception as e:
        st.error(f"Train-test split error: {e}")
        st.stop()

    # Regression
    if model_type == "Regression":
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target must be numeric.")
            st.stop()

        model = RandomForestRegressor()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("Regression Results")
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("RMSE:", mean_squared_error(y_test, preds)**0.5)

    # Classification
    else:
        if pd.api.types.is_numeric_dtype(y):
            st.error("Target must be categorical.")
            st.stop()

        model = RandomForestClassifier()
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        st.subheader("Classification Results")
        st.write("Accuracy:", accuracy_score(y_test, preds))

        st.success("Model trained successfully!")
