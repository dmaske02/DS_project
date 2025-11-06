import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Census Data EDA App", layout="wide")

st.title("📊 Census Household Amenities EDA Dashboard")

st.write("""
This app performs **Exploratory Data Analysis (EDA)** on the uploaded dataset.
Upload your CSV file to view missing values, correlations, and key statistics.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    # Dataset info
    st.subheader("Basic Information")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns:**", list(df.columns))

    # Missing values
    st.subheader("Missing Value Summary")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    }).sort_values(by='Missing Values', ascending=False)
    st.dataframe(missing_df)

    # Drop missing rows for analysis
    df_clean = df.dropna()

    st.subheader("Cleaned Data Shape")
    st.write(f"Remaining rows after dropping missing values: {len(df_clean)}")

    # Correlation heatmap (Top 30 most varying features)
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        top30_cols = numeric_df.std().sort_values(ascending=False).head(30).index
        corr_matrix = numeric_df[top30_cols].corr()

        st.subheader("Correlation Heatmap (Top 30 Features)")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5, square=True)
        st.pyplot(fig)

        # Top positive and negative correlations
        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]

        corr_pairs['Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Feature 1'], row['Feature 2']])), axis=1)
        corr_pairs = corr_pairs.drop_duplicates(subset='Pair').drop(columns='Pair')

        top_pos = corr_pairs[corr_pairs['Correlation'] > 0].sort_values(by='Correlation', ascending=False).head(10)
        top_neg = corr_pairs[corr_pairs['Correlation'] < 0].sort_values(by='Correlation').head(10)

        st.subheader("Top 10 Positive Correlations")
        st.dataframe(top_pos)

        st.subheader("Top 10 Negative Correlations")
        st.dataframe(top_neg)
    else:
        st.warning("No numeric columns found for correlation analysis.")
else:
    st.info("Please upload a CSV file to begin.")
