# ------------------------------------------------------
# SIMPLE + CLEAN STREAMLIT DASHBOARD FOR ANY DATASET
# ------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Simple Census Data Dashboard", layout="wide")

# ------------------------------------------------------
# Page Title
# ------------------------------------------------------
st.title("📊 Simple Data Exploration Dashboard")

st.write("""
This is a **clean and simple Streamlit dashboard**  
to explore your dataset without complexity.
""")

# ------------------------------------------------------
# Upload Section
# ------------------------------------------------------
uploaded_file = st.file_uploader("📥 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your dataset to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------------------------------
# Dataset Preview
# ------------------------------------------------------
st.header("📁 Dataset Preview")
st.dataframe(df.head())

# Basic shape info
st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")

numerics = df.select_dtypes(include=['float64','int64']).columns.tolist()
categoricals = df.select_dtypes(include=['object','category']).columns.tolist()

# ------------------------------------------------------
# Missing Values
# ------------------------------------------------------
st.header("❗ Missing Values")
missing = df.isnull().sum()
missing_df = missing[missing > 0]

if missing_df.empty:
    st.success("No missing values found.")
else:
    st.dataframe(missing_df)

# ------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------
st.header("📊 Summary Statistics")
if numerics:
    st.subheader("Numeric Columns")
    st.write(df[numerics].describe())
else:
    st.write("No numeric columns found.")

# ------------------------------------------------------
# Distribution Plot
# ------------------------------------------------------
if numerics:
    st.header("📈 Distribution Plot")
    col = st.selectbox("Choose a numeric column", numerics)

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

# ------------------------------------------------------
# Categorical Counts
# ------------------------------------------------------
if categoricals:
    st.header("📊 Categorical Value Counts")
    col_cat = st.selectbox("Choose a categorical column", categoricals)

    fig, ax = plt.subplots()
    df[col_cat].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Value counts of {col_cat}")
    st.pyplot(fig)

# ------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------
st.header("🔥 Correlation Heatmap")

if len(numerics) >= 2:
    corr = df[numerics].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.info("Need at least 2 numeric columns to show correlation heatmap.")

# ------------------------------------------------------
# END
# ------------------------------------------------------
st.success("Dashboard Loaded Successfully!")
