# 🏠 Census Household Amenities EDA Dashboard

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-red?logo=streamlit)](https://dsproject-fxdyxdxyrmntyfbvza3kxe.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-green)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## 📊 Overview

The **Census Household Amenities EDA Dashboard** is an interactive web app built with **Streamlit** that enables users to perform **Exploratory Data Analysis (EDA)** on census or household-level datasets.

It helps you:
- Upload your own CSV dataset  
- Explore missing values  
- View data structure and summary  
- Visualize correlations between numerical features  
- Discover insights through an interactive dashboard  

🔗 **Live Demo:**  
👉 [Click here to open the app](https://dsproject-fxdyxdxyrmntyfbvza3kxe.streamlit.app/)

---

## 🚀 Features

✨ Upload your dataset (`.csv` format)  
📋 View dataset shape, columns, and sample records  
📉 Analyze missing value distribution  
🧹 Automatically clean and preprocess missing data  
🔥 Visualize **top 30 correlated numeric features** using a heatmap  
📈 View **top 10 positive** and **top 10 negative** correlations  
🎨 Clean, simple, and fully interactive Streamlit UI  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Language | Python 3 |

---

## 📂 Project Structure

📁 census-eda-dashboard/
│
├── app.py # Streamlit application script
├── requirements.txt # List of dependencies
├── Ds assignment.ipynb # Original Jupyter notebook for EDA
├── README.md # Project documentation
└── (optional) sample.csv # Example dataset (if included)


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/census-eda-dashboard.git
cd census-eda-dashboard

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app locally
streamlit run app.py


The app will open automatically in your browser.

☁️ Deployment on Streamlit Cloud

You can easily deploy your own version:

Push this project to GitHub

Go to Streamlit Cloud

Click "New App"

Select your GitHub repo and set the entry point as app.py

Click Deploy — your app will be live in seconds 🚀
💡 Future Enhancements

Add downloadable summary report (PDF/CSV)

Include more visualizations (pairplot, distribution plots, boxplots)

Add outlier and skewness detection

Implement feature importance ranking
