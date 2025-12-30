import os
import streamlit as st
import pandas as pd
import joblib

# BASE_DIR sử dụng os.path
BASE_DIR = os.path.join(os.getcwd(), "demo")

# Load model với try/except để bắt lỗi
try:
    kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    cluster_profile = joblib.load(os.path.join(BASE_DIR, "cluster_profile.pkl"))
except Exception as e:
    st.error(f"Lỗi khi load model/scaler: {e}")
    st.stop()


st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("Customer Segmentation App")
st.write("Enter customer information to predict customer segment.")

# ===============================
# Input fields
# ===============================
income = st.number_input("Income", min_value=0, max_value=120000, value=52000, step=1000)
num_deals_purchases = st.number_input("Number of Deal Purchases", min_value=0, max_value=6, value=2)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=12, value=4)
num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, max_value=10, value=2)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=13, value=5)
num_web_visits_month = st.number_input("Number of Web Visits per Month", min_value=0, max_value=13, value=6)
total_children = st.number_input("Total Children", min_value=0, max_value=3, value=1)
total_spending = st.number_input("Total Spending", min_value=0, max_value=2200, value=400, step=50)

education = st.selectbox("Education", ["Graduate", "Undergraduate", "Postgraduate"])

# ===============================
# Prepare input dataframe theo thứ tự X
# ===============================
X = [
    'Income',
    'NumDealsPurchases',
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases',
    'NumWebVisitsMonth',
    'TotalChildren',
    'TotalSpending',
    'Education_Postgraduate',
    'Education_Undergraduate'
]

input_data = pd.DataFrame([[
    income,
    num_deals_purchases,
    num_web_purchases,
    num_catalog_purchases,
    num_store_purchases,
    num_web_visits_month,
    total_children,
    total_spending,
    1 if education == "Postgraduate" else 0,
    1 if education == "Undergraduate" else 0
]], columns=X)

# ===============================
# Scale input
# ===============================
input_scaled = scaler.transform(input_data)

# ===============================
# Predict
# ===============================
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment: Cluster {cluster}")

    st.subheader("Cluster Profile (Average Values)")
    st.dataframe(cluster_profile.loc[[cluster]])
