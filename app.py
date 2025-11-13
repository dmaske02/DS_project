# app.py (Fixed: no-plotly + safe multiselect defaults)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Census Data: EDA + ML (Fixed)", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_numeric_and_categorical(df):
    numerics = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categoricals = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerics, categoricals

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')
    return preprocessor

# -------------------------
# App UI - upload
# -------------------------
st.title("📊 Census Data: EDA + ML Dashboard (Fixed)")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
use_demo = st.button("Use demo dataset (synthetic)")

if uploaded is None and not use_demo:
    st.info("Upload a CSV or use the demo dataset to continue.")
    st.stop()

# -------------------------
# Load Data
# -------------------------
if use_demo and uploaded is None:
    np.random.seed(0)
    df = pd.DataFrame({
        'household_size': np.random.randint(1, 8, size=500),
        'num_rooms': np.random.randint(1, 6, size=500),
        'has_electricity': np.random.choice(['yes','no'], size=500, p=[0.85,0.15]),
        'has_toilet': np.random.choice(['yes','no'], size=500, p=[0.7,0.3]),
        'distance_to_market_km': np.round(np.random.exponential(1.5, size=500), 2),
        'income_monthly': np.random.normal(12000, 4000, size=500).clip(2000, 50000)
    })
    st.success("Demo dataset loaded.")
else:
    df = pd.read_csv(uploaded)
    st.success("File uploaded.")

df_original = df.copy()
numerics, categoricals = get_numeric_and_categorical(df)

# -------------------------
# Layout Tabs
# -------------------------
tabs = st.tabs(["Overview", "EDA", "Model Training", "Feature Importance", "Predict", "Download / About"])

# -------------------------
# Tab 1: Overview & Cleaning
# -------------------------
with tabs[0]:
    st.header("Overview & Cleaning")
    st.subheader("Preview")
    st.dataframe(df.head(10))

    st.write("Shape:", df.shape)
    st.write("Numeric columns:", numerics)
    st.write("Categorical columns:", categoricals)

    st.subheader("Missing values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])

    st.subheader("Cleaning options")
    clean_choice = st.selectbox("How to handle missing values?", ["Do nothing", "Drop rows with any NA", "Impute (numeric=median, categorical=mode)"])
    if st.button("Apply cleaning"):
        if clean_choice == "Drop rows with any NA":
            df = df.dropna().reset_index(drop=True)
            st.success("Dropped rows with NA.")
        elif clean_choice == "Impute (numeric=median, categorical=mode)":
            for col in numerics:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            for col in categoricals:
                if col in df.columns:
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
                    else:
                        df[col] = df[col].fillna("missing")
            st.success("Imputation done.")
        else:
            st.info("No cleaning applied.")
        # update lists
        numerics, categoricals = get_numeric_and_categorical(df)
        st.write("Updated numeric columns:", numerics)
        st.write("Updated categorical columns:", categoricals)

# -------------------------
# Tab 2: EDA
# -------------------------
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")

    if numerics:
        st.subheader("Numeric distribution")
        num_col = st.selectbox("Select numeric column", numerics)
        fig, ax = plt.subplots()
        sns.histplot(df[num_col].dropna(), kde=True, ax=ax)
        ax.set_xlabel(num_col)
        st.pyplot(fig)
        st.write(df[num_col].describe())
    else:
        st.warning("No numeric columns available for distribution plots.")

    if categoricals:
        st.subheader("Categorical counts")
        cat_col = st.selectbox("Select categorical column", categoricals)
        fig, ax = plt.subplots()
        df[cat_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Counts of {cat_col}")
        st.pyplot(fig)
    else:
        st.info("No categorical columns available for count plots.")

    if len(numerics) >= 2:
        st.subheader("Correlation heatmap")
        # let user choose how many top numeric features by variance
        max_show = min(20, len(numerics))
        top_n = st.slider("Number of numeric features to include (by variance)", min_value=2, max_value=max_show, value=min(8, max_show))
        top_feats = df[numerics].var().sort_values(ascending=False).head(top_n).index.tolist()
        corr = df[top_feats].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns for correlation heatmap.")

# -------------------------
# Tab 3: Model Training
# -------------------------
with tabs[2]:
    st.header("Model Training & Evaluation")

    if not numerics:
        st.error("No numeric columns available — at least one numeric target is required for modeling.")
    else:
        target = st.selectbox("Select target column (numeric)", numerics, index=len(numerics)-1)

        # build safe available_features list and safe default_features
        available_features = [c for c in df.columns if c != target]
        # default features: numeric columns excluding target
        default_features = [c for c in numerics if c != target and c in available_features]
        # fallback default: first up to 4 available features if default_features empty
        if not default_features:
            default_features = available_features[:4]

        features = st.multiselect("Select feature columns", available_features, default=default_features)

        if not features:
            st.info("Select at least one feature to enable training.")
        else:
            test_size = st.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2)
            random_state = st.number_input("Random state (int)", value=42, step=1)

            if st.button("Train models"):
                X = df[features].copy()
                y = df[target].copy()

                # detect numeric / categorical in selected features
                sel_numerics = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                sel_categoricals = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                if len(sel_numerics) + len(sel_categoricals) == 0:
                    st.error("Selected features have no usable columns (they may be all missing or unsupported dtypes).")
                else:
                    preprocessor = build_preprocessor(sel_numerics, sel_categoricals)

                    pipelines = {
                        "LinearRegression": Pipeline([('pre', preprocessor), ('model', LinearRegression())]),
                        "Ridge": Pipeline([('pre', preprocessor), ('model', Ridge(alpha=1.0))]),
                        "RandomForest": Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=int(random_state), n_jobs=-1))])
                    }

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

                    results = []
                    trained_models = {}
                    for name, pipe in pipelines.items():
                        with st.spinner(f"Training {name} ..."):
                            try:
                                pipe.fit(X_train, y_train)
                                y_pred = pipe.predict(X_test)
                                trmse = rmse(y_test, y_pred)
                                tr2 = r2_score(y_test, y_pred)
                                results.append({"model": name, "test_RMSE": round(trmse, 4), "test_R2": round(tr2, 4)})
                                trained_models[name] = pipe
                            except Exception as e:
                                st.error(f"Training {name} failed: {e}")

                    if results:
                        res_df = pd.DataFrame(results).sort_values("test_RMSE")
                        st.subheader("Model comparison (by test RMSE)")
                        st.dataframe(res_df)
                        # save in session
                        st.session_state['trained_models'] = trained_models
                        st.session_state['selected_features'] = features
                        st.session_state['target'] = target
                        st.success("Training finished and models saved in session state.")
                    else:
                        st.warning("No models finished training successfully.")

# -------------------------
# Tab 4: Feature Importance
# -------------------------
with tabs[3]:
    st.header("Feature Importance")
    if 'trained_models' not in st.session_state:
        st.info("Train models in the 'Model Training' tab first.")
    else:
        trained = st.session_state['trained_models']
        sel_feats = st.session_state['selected_features']

        if "RandomForest" in trained:
            rf_pipe = trained["RandomForest"]
            try:
                rf_model = rf_pipe.named_steps['model']
                # try to assemble feature names after one-hot
                pre = rf_pipe.named_steps['pre']
                num_feats = pre.transformers_[0][2] if pre.transformers_[0][2] is not None else []
                cat_feats = pre.transformers_[1][2] if pre.transformers_[1][2] is not None else []
                ohe = pre.transformers_[1][1].named_steps['onehot']
                try:
                    cat_ohe_names = ohe.get_feature_names_out(cat_feats)
                except Exception:
                    # fallback: use categorical feature names as-is
                    cat_ohe_names = cat_feats
                feature_names = list(num_feats) + list(cat_ohe_names)
                importances = rf_model.feature_importances_
                fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                fi_df = fi_df.sort_values('importance', ascending=False).head(40)
                st.subheader("Random Forest importances")
                st.dataframe(fi_df.reset_index(drop=True))
                fig, ax = plt.subplots(figsize=(6, min(0.4 * len(fi_df), 8)))
                sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error("Could not compute RandomForest importances: " + str(e))
        else:
            st.info("RandomForest not trained or unavailable.")

        # linear coefficients (if linear trained)
        if "LinearRegression" in trained:
            st.subheader("Linear model coefficients (approximate)")
            try:
                lr_pipe = trained["LinearRegression"]
                lr_model = lr_pipe.named_steps['model']
                pre = lr_pipe.named_steps['pre']
                num_feats = pre.transformers_[0][2] if pre.transformers_[0][2] is not None else []
                cat_feats = pre.transformers_[1][2] if pre.transformers_[1][2] is not None else []
                ohe = pre.transformers_[1][1].named_steps['onehot']
                try:
                    cat_ohe_names = ohe.get_feature_names_out(cat_feats)
                except Exception:
                    cat_ohe_names = cat_feats
                feature_names = list(num_feats) + list(cat_ohe_names)
                coefs = lr_model.coef_
                coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
                coef_df['abs_coef'] = coef_df['coefficient'].abs()
                coef_df = coef_df.sort_values('abs_coef', ascending=False).head(40)
                st.dataframe(coef_df[['feature', 'coefficient']].reset_index(drop=True))
                fig, ax = plt.subplots(figsize=(6, min(0.4 * len(coef_df), 8)))
                sns.barplot(x='coefficient', y='feature', data=coef_df.sort_values('coefficient'), ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error("Could not extract linear coefficients: " + str(e))

# -------------------------
# Tab 5: Predict
# -------------------------
with tabs[4]:
    st.header("Prediction (Interactive)")

    if 'trained_models' not in st.session_state:
        st.info("Train models first to use prediction.")
    else:
        models = st.session_state['trained_models']
        available_models = list(models.keys())
        model_choice = st.selectbox("Choose model for prediction", available_models)

        sel_feats = st.session_state.get('selected_features', [])
        if not sel_feats:
            st.warning("No selected features available for prediction.")
        else:
            # Build user input form
            user_inputs = {}
            for feat in sel_feats:
                if feat in numerics:
                    # numeric
                    default_val = float(df[feat].median()) if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]) else 0.0
                    user_inputs[feat] = st.number_input(f"{feat} (numeric)", value=default_val, format="%f")
                else:
                    # categorical
                    if feat in df.columns:
                        opts = sorted(df[feat].dropna().unique().tolist())
                        if not opts:
                            user_inputs[feat] = st.text_input(f"{feat} (categorical)", value="")
                        else:
                            user_inputs[feat] = st.selectbox(f"{feat} (categorical)", options=opts)
                    else:
                        user_inputs[feat] = st.text_input(f"{feat} (categorical)", value="")

            if st.button("Predict"):
                input_df = pd.DataFrame([user_inputs])
                try:
                    pred = models[model_choice].predict(input_df)[0]
                    st.success(f"Predicted {st.session_state.get('target', 'target')}: {pred}")
                    if st.checkbox("Save chosen model to joblib"):
                        fname = f"model_{model_choice}.joblib".replace(" ", "_")
                        joblib.dump(models[model_choice], fname)
                        st.write(f"Saved model to `{fname}`")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# -------------------------
# Tab 6: Download / About
# -------------------------
with tabs[5]:
    st.header("Download & About")
    st.subheader("Download current dataset (CSV)")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, "cleaned_data.csv", mime="text/csv")

    st.markdown("""
    **About**
    - This app provides EDA and simple ML modeling for census/household datasets.
    - If you deployed on Streamlit Cloud, ensure `requirements.txt` includes `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`, and `joblib`.
    - If you still face errors, check your app logs in Streamlit Cloud's "Manage app" → "Logs".
    """)
