# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import io

st.set_page_config(page_title="Census EDA & Modeling", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def summarize_df(df):
    desc = df.describe(include='all').T
    desc['n_missing'] = df.isnull().sum()
    desc['pct_missing'] = (desc['n_missing'] / len(df)) * 100
    return desc

def get_numeric_and_categorical(df):
    numerics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricals = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerics, categoricals

def build_preprocessor(numeric_features, categorical_features, impute_strategy='median'):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
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

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# -------------------------
# Title & upload
# -------------------------
st.title("📊 Census Household Amenities — EDA, Modeling & Prediction")
st.write("Upload your CSV (household/census dataset). The app will let you explore, train models, compare them, and make predictions.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
sample_data_btn = st.button("Use sample synthetic dataset (demo)")

if uploaded is None and not sample_data_btn:
    st.info("Upload a CSV or click the demo button to continue.")
    st.stop()

# -------------------------
# Load data
# -------------------------
if sample_data_btn and uploaded is None:
    # build a small synthetic dataset for demo
    np.random.seed(0)
    df = pd.DataFrame({
        'household_size': np.random.randint(1, 8, size=500),
        'num_rooms': np.random.randint(1, 6, size=500),
        'has_electricity': np.random.choice(['yes','no'], size=500, p=[0.85,0.15]),
        'has_toilet': np.random.choice(['yes','no'], size=500, p=[0.7,0.3]),
        'distance_to_market_km': np.round(np.random.exponential(1.5, size=500), 2),
        'income_monthly': np.random.normal(12000, 4000, size=500).clip(2000, 50000)
    })
    st.success("Loaded demo dataset.")
else:
    df = pd.read_csv(uploaded)
    st.success("File uploaded.")

# keep original copy
df_original = df.copy()

# -------------------------
# Multi-page like tabs
# -------------------------
pages = ["1. Overview & Cleaning", "2. EDA", "3. Modeling & Comparison", "4. Feature Importance", "5. Predict", "6. Download / About"]
page = st.radio("Choose view", pages, index=0, horizontal=True)

# global selections used across pages
numerics, categoricals = get_numeric_and_categorical(df)

# -------------------------
# Page 1: Overview & Cleaning
# -------------------------
if page == pages[0]:
    st.header("1) Overview & Cleaning")
    st.subheader("Preview")
    st.dataframe(df.head(10))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"Rows: **{df.shape[0]}**")
        st.write(f"Columns: **{df.shape[1]}**")
    with col2:
        st.write("Numeric columns: " + ", ".join(numerics) if numerics else "Numeric columns: None")
        st.write("Categorical columns: " + ", ".join(categoricals) if categoricals else "Categorical columns: None")

    st.subheader("Summary Statistics")
    desc = summarize_df(df)
    st.dataframe(desc)

    st.subheader("Missing values visual")
    missing = df.isnull().sum().sort_values(ascending=False)
    st.bar_chart(missing[missing > 0])

    st.write("---")
    st.subheader("Cleaning options")
    clean_choice = st.selectbox("How would you like to handle missing values?", ["Drop rows with any NA", "Impute (numeric: median, categorical: mode)"])
    if st.button("Apply cleaning"):
        if clean_choice == "Drop rows with any NA":
            df = df.dropna().reset_index(drop=True)
            st.success(f"Dropped NA rows → new shape {df.shape}")
        else:
            # perform simple imputation inplace for display (not yet preprocessor)
            num_imputer = SimpleImputer(strategy='median')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            if numerics:
                df[numerics] = num_imputer.fit_transform(df[numerics])
            if categoricals:
                df[categoricals] = cat_imputer.fit_transform(df[categoricals])
            st.success("Imputed missing values (numeric=median, categorical=mode).")
        # update derived lists
        numerics, categoricals = get_numeric_and_categorical(df)
        st.write("Updated numeric columns:", numerics)
        st.write("Updated categorical columns:", categoricals)

# -------------------------
# Page 2: EDA
# -------------------------
elif page == pages[1]:
    st.header("2) Exploratory Data Analysis (EDA)")
    st.markdown("Use the controls to explore numeric & categorical features.")

    # quick distribution
    if numerics:
        st.subheader("Numeric Distributions")
        col_n = st.selectbox("Select numeric column for distribution/boxplot", numerics, index=0)
        st.plotly_chart(px.histogram(df, x=col_n, marginal="box", nbins=40), use_container_width=True)
        # describe
        st.write(df[col_n].describe())

        # outliers via IQR
        q1 = df[col_n].quantile(0.25)
        q3 = df[col_n].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = df[(df[col_n] < lower) | (df[col_n] > upper)].shape[0]
        st.write(f"Estimated outliers (IQR method): {n_outliers} rows")

    else:
        st.warning("No numeric columns available for numeric EDA.")

    # categoricals
    if categoricals:
        st.subheader("Categorical distributions")
        cat_col = st.selectbox("Select categorical column", categoricals, index=0)
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, 'count']
        st.plotly_chart(px.bar(vc, x=cat_col, y='count', text='count'), use_container_width=True)
        st.write(vc)
    else:
        st.info("No categorical columns detected.")

    # correlation heatmap for numeric columns
    if len(numerics) >= 2:
        st.subheader("Correlation Heatmap (numeric features)")
        top_n = st.slider("Max features to show (by variance)", min_value=4, max_value=min(40, len(numerics)), value=min(12, len(numerics)))
        # sort numeric features by variance
        top_feats = df[numerics].var().sort_values(ascending=False).head(top_n).index.tolist()
        corr = df[top_feats].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns for correlation heatmap.")

# -------------------------
# Page 3: Modeling & Comparison
# -------------------------
elif page == pages[2]:
    st.header("3) Modeling & Comparison")
    st.markdown("Select a target column (numeric) and features to train models. Models: LinearRegression, Ridge, RandomForest.")

    # target selection
    if not numerics:
        st.error("No numeric column available to use as target. Add/convert a numeric target and retry.")
        st.stop()
    target = st.selectbox("Select target (numeric)", numerics, index=len(numerics)-1)

    # feature selection
    possible_features = [c for c in df.columns if c != target]
    st.write("Available features (exclude target):", possible_features)
    selected_features = st.multiselect("Select features to use (prefer numeric + low-cardinality categoricals)", possible_features, default=[c for c in possible_features if c in numerics][:4])

    if not selected_features:
        st.info("Select at least one feature to train.")
        st.stop()

    test_size = st.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.2)
    random_state = st.number_input("Random state (int)", value=42, step=1)
    cv_folds = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5)

    # prepare data
    X = df[selected_features].copy()
    y = df[target].copy()

    # detect selected numeric & categorical
    sel_numerics = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    sel_categoricals = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    preprocessor = build_preprocessor(sel_numerics, sel_categoricals)
    # pipelines
    pipelines = {
        "LinearRegression": Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())]),
        "Ridge": Pipeline(steps=[('pre', preprocessor), ('model', Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1))])
    }

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    if st.button("Train & Evaluate Models"):
        results = []
        trained_models = {}
        for name, pipe in pipelines.items():
            with st.spinner(f"Training {name} ..."):
                # cross val rmse
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                neg_mse_scores = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
                cv_rmse = np.mean(np.sqrt(-neg_mse_scores))
                # fit full pipeline
                pipe.fit(X_train, y_train)
                y_pred_test = pipe.predict(X_test)
                test_rmse = rmse(y_test, y_pred_test)
                test_r2 = r2_score(y_test, y_pred_test)
                results.append({
                    "model": name,
                    "cv_RMSE": round(cv_rmse, 4),
                    "test_RMSE": round(test_rmse, 4),
                    "test_R2": round(test_r2, 4)
                })
                trained_models[name] = pipe

        results_df = pd.DataFrame(results).sort_values("test_RMSE")
        st.subheader("Model comparison")
        st.table(results_df)

        # store models in session_state for other pages
        st.session_state['trained_models'] = trained_models
        st.session_state['selected_features'] = selected_features
        st.session_state['target'] = target
        st.success("Models trained and saved in session.")

        # plot predicted vs actual for best model
        best_model_name = results_df.iloc[0]['model']
        best_pipe = trained_models[best_model_name]
        y_pred_best = best_pipe.predict(X_test)
        fig = px.scatter(x=y_test, y=y_pred_best, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"Predicted vs Actual — {best_model_name}")
        fig.add_shape(type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(), line=dict(dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Page 4: Feature Importance
# -------------------------
elif page == pages[3]:
    st.header("4) Feature Importance & Coefficients")
    if 'trained_models' not in st.session_state:
        st.info("Train models in the Modeling page first.")
        st.stop()

    models = st.session_state['trained_models']
    sel_feats = st.session_state['selected_features']

    st.write("Selected features used during training:")
    st.write(sel_feats)

    # For linear models show coefficients (after preprocessing)
    st.subheader("Linear model coefficients (approximate after preprocessing)")
    if "LinearRegression" in models:
        lr_pipe = models['LinearRegression']
        # get preprocessor and model
        pre = lr_pipe.named_steps['pre']
        model = lr_pipe.named_steps['model']
        # Need to map feature names after OneHotEncoder -> get feature names
        # OneHotEncoder present in ColumnTransformer: attempt to extract names
        try:
            # numeric names
            num_feats = pre.transformers_[0][2]  # numeric cols
            cat_transformer = pre.transformers_[1][1].named_steps['onehot']
            cat_feats = pre.transformers_[1][2]
            ohe_feature_names = cat_transformer.get_feature_names_out(cat_feats)
            feature_names = list(num_feats) + list(ohe_feature_names)
            coefs = model.coef_
            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
            coef_df['abs_coef'] = coef_df['coefficient'].abs()
            coef_df = coef_df.sort_values('abs_coef', ascending=False).head(40)
            st.dataframe(coef_df[['feature', 'coefficient']].reset_index(drop=True))
            # plot
            fig = px.bar(coef_df.sort_values('coefficient'), x='coefficient', y='feature', orientation='h', title="Top linear coefficients")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Could not extract exact feature names after preprocessing: " + str(e))

    # For RandomForest show feature importances (via pipeline)
    st.subheader("Random Forest feature importances")
    if "RandomForest" in models:
        rf_pipe = models['RandomForest']
        rf = rf_pipe.named_steps['model']
        pre = rf_pipe.named_steps['pre']
        # try to recover feature names after one-hot
        try:
            num_feats = pre.transformers_[0][2]
            cat_feats = pre.transformers_[1][2]
            ohe = pre.transformers_[1][1].named_steps['onehot']
            cat_ohe_names = ohe.get_feature_names_out(cat_feats)
            final_feature_names = list(num_feats) + list(cat_ohe_names)
            importances = rf.feature_importances_
            fi_df = pd.DataFrame({'feature': final_feature_names, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=False).head(40)
            st.dataframe(fi_df.reset_index(drop=True))
            fig = px.bar(fi_df.sort_values('importance'), x='importance', y='feature', orientation='h', title='Top RandomForest importances')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Could not extract feature names after preprocessing: " + str(e))

# -------------------------
# Page 5: Predict
# -------------------------
elif page == pages[4]:
    st.header("5) Make Predictions (Interactive)")
    if 'trained_models' not in st.session_state:
        st.info("Train models on the Modeling page first.")
        st.stop()

    models = st.session_state['trained_models']
    sel_feats = st.session_state['selected_features']
    target = st.session_state['target']

    st.write("Models available:", list(models.keys()))
    model_choice = st.selectbox("Choose model for prediction", list(models.keys()))

    st.subheader("Enter feature values (leave blank to use dataset mean/mode)")
    user_input = {}
    input_cols = []
    # for each selected feature, create an input field
    for feat in sel_feats:
        dtype = df_original[feat].dtype if feat in df_original.columns else df[feat].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            val = st.number_input(f"{feat} (numeric)", value=float(df[feat].median()) if feat in df.columns else 0.0, format="%f")
            user_input[feat] = val
        else:
            # get unique values
            unique_vals = df[feat].dropna().unique().tolist() if feat in df.columns else []
            if len(unique_vals) == 0:
                txt = st.text_input(f"{feat} (categorical)", value="")
                user_input[feat] = txt
            else:
                opt = st.selectbox(f"{feat} (categorical)", ["<use default>"] + unique_vals)
                if opt == "<use default>":
                    # default to most frequent
                    user_input[feat] = df[feat].mode().iloc[0] if feat in df.columns and not df[feat].mode().empty else unique_vals[0]
                else:
                    user_input[feat] = opt

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        model_pipe = models[model_choice]
        try:
            pred = model_pipe.predict(input_df)[0]
            st.success(f"Predicted {target}: **{pred:.4f}**")
            # Save a joblib copy of model if user wants
            if st.checkbox("Save model to file (joblib)"):
                buf = io.BytesIO()
                joblib.dump(model_pipe, "trained_model.joblib")
                st.write("Model saved as `trained_model.joblib` in working directory.")
        except Exception as e:
            st.error("Prediction failed. Reason: " + str(e))

# -------------------------
# Page 6: Download / About
# -------------------------
elif page == pages[5]:
    st.header("6) Download & About")
    st.subheader("Download cleaned data")
    # allow user to download cleaned dataset (after any cleaning done earlier)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download current dataset (CSV)", csv_bytes, file_name="cleaned_census_data.csv", mime="text/csv")

    st.write("---")
    st.markdown("""
    **About this app**

    - Built for quick EDA + baseline modeling on census/household datasets.
    - Models included: Linear Regression, Ridge Regression, Random Forest Regressor.
    - Preprocessing: imputation (median/mode), standard scaling, one-hot encoding for categoricals.
    - After training, models are stored in session and can be used for predictions.

    **Next enhancements you can ask for**
    - Grid search / hyperparameter tuning with `RandomizedSearchCV` or `Optuna`.
    - Automatic feature engineering (target encoding, interactions).
    - Multi-target regression support.
    - Streamlit multipage file structure (separate files for pages).
    """)

