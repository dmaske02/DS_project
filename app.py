# app.py (No Plotly Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import io

st.set_page_config(page_title="Census EDA & ML Dashboard", layout="wide")

# -------------------------------------------
# Helper Functions
# -------------------------------------------
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def get_numeric_and_categorical(df):
    num = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return num, cat

def build_preprocessor(numeric, categorical):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric),
        ('cat', categorical_pipeline, categorical)
    ])
    return preprocessor

# -------------------------------------------
# Upload
# -------------------------------------------
st.title("📊 Census Data: EDA + ML Dashboard (No Plotly)")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded is None:
    st.info("📥 Upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded)
df_original = df.copy()

numerics, categoricals = get_numeric_and_categorical(df)

# -------------------------------------------
# Tabs
# -------------------------------------------
tabs = st.tabs(["Overview", "EDA", "Model Training", "Feature Importance", "Predict", "Download"])

# ===========================================
# TAB 1 — OVERVIEW
# ===========================================
with tabs[0]:
    st.header("Dataset Overview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Numeric Columns:", numerics)
    st.write("Categorical Columns:", categoricals)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Clean Missing Data")
    choice = st.radio("Choose method:", ["Drop NA rows", "Impute (median/mode)"])

    if st.button("Apply Cleaning"):
        if choice == "Drop NA rows":
            df = df.dropna()
        else:
            for col in numerics:
                df[col] = df[col].fillna(df[col].median())
            for col in categoricals:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Cleaning complete!")
        numerics, categoricals = get_numeric_and_categorical(df)

# ===========================================
# TAB 2 — EDA
# ===========================================
with tabs[1]:
    st.header("Exploratory Data Analysis")

    # -----------------
    # Numeric Distribution
    # -----------------
    if numerics:
        st.subheader("Distribution Plot")
        col = st.selectbox("Select numeric column", numerics)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        st.write(df[col].describe())
    else:
        st.warning("No numeric columns available.")

    # -----------------
    # Categorical count
    # -----------------
    if categoricals:
        st.subheader("Categorical Count Plot")
        cat = st.selectbox("Select categorical column", categoricals)

        fig, ax = plt.subplots()
        df[cat].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Counts of {cat}")
        st.pyplot(fig)
    else:
        st.info("No categorical columns available.")

    # -----------------
    # Correlation Heatmap
    # -----------------
    if len(numerics) >= 2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numerics].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ===========================================
# TAB 3 — MODEL TRAINING
# ===========================================
with tabs[2]:
    st.header("Train ML Models")

    if not numerics:
        st.error("Need numeric target.")
        st.stop()

    target = st.selectbox("Select target column (numeric only)", numerics)
    features = st.multiselect("Select feature columns", [c for c in df.columns if c != target], default=numerics)

    if st.button("Train"):
        X = df[features]
        y = df[target]

        num_sel, cat_sel = get_numeric_and_categorical(X)
        preproc = build_preprocessor(num_sel, cat_sel)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=200)
        }

        results = []
        trained = {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        for name, model in models.items():
            pipe = Pipeline([('pre', preproc), ('model', model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            trained[name] = pipe
            results.append({
                "Model": name,
                "RMSE": rmse(y_test, preds),
                "R2": r2_score(y_test, preds)
            })

        res_df = pd.DataFrame(results)
        st.write("### Results")
        st.dataframe(res_df)

        # Store models
        st.session_state["trained_models"] = trained
        st.session_state["selected_features"] = features
        st.session_state["target"] = target

# ===========================================
# TAB 4 — FEATURE IMPORTANCE
# ===========================================
with tabs[3]:
    st.header("Feature Importance")

    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
        st.stop()

    models = st.session_state["trained_models"]
    target = st.session_state["target"]
    features = st.session_state["selected_features"]

    # -----------------
    # Random Forest Importances
    # -----------------
    if "Random Forest" in models:
        rf = models["Random Forest"]
        model = rf.named_steps["model"]

        st.subheader("Random Forest Feature Importances")

        try:
            importances = model.feature_importances_

            fig, ax = plt.subplots()
            sns.barplot(x=importances, y=features, ax=ax)
            st.pyplot(fig)
        except:
            st.error("Couldn't extract feature importances.")

# ===========================================
# TAB 5 — PREDICTION
# ===========================================
with tabs[4]:
    st.header("Make Predictions")

    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
        st.stop()

    models = st.session_state["trained_models"]
    model_choice = st.selectbox("Select model", list(models.keys()))

    inputs = {}
    for col in st.session_state["selected_features"]:
        if col in numerics:
            inputs[col] = st.number_input(col, value=float(df[col].median()))
        else:
            inputs[col] = st.selectbox(col, sorted(df[col].unique()))

    if st.button("Predict"):
        user_df = pd.DataFrame([inputs])
        model = models[model_choice]
        pred = model.predict(user_df)[0]
        st.success(f"Predicted {st.session_state['target']}: {pred}")

# ===========================================
# TAB 6 — DOWNLOAD
# ===========================================
with tabs[5]:
    st.header("Download Data & Models")

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv")

    if "trained_models" in st.session_state:
        if st.button("Save Random Forest Model"):
            joblib.dump(st.session_state["trained_models"]["Random Forest"], "rf_model.joblib")
            st.success("Saved as rf_model.joblib")
