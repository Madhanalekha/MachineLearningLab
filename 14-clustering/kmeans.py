import os

# Fix Streamlit watcher issue
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import pandas as pd
import numpy as np
import kagglehub

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Title
# -------------------------------
st.title("Customer Segmentation using K-Means Clustering")

# -------------------------------
# Download Dataset from Kaggle
# -------------------------------
@st.cache_data
def load_data():

    path = kagglehub.dataset_download("shwetabh123/mall-customers")

    df = pd.read_csv(os.path.join(path, "Mall_Customers.csv"))

    return df


df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------------
# Feature Selection
# -------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train K-Means Model
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# -------------------------------
# User Input
# -------------------------------
st.subheader("Enter Customer Details")

income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Cluster"):

    user_data = np.array([[income, spending]])

    user_scaled = scaler.transform(user_data)

    cluster = kmeans.predict(user_scaled)[0]

    st.success(f"The customer belongs to Cluster: {cluster}")

    # Interpretation
    if cluster == 0:
        st.info("Segment: Low Income - Low Spending Customer")

    elif cluster == 1:
        st.info("Segment: High Income - High Spending Customer")

    elif cluster == 2:
        st.info("Segment: High Income - Low Spending Customer")

    elif cluster == 3:
        st.info("Segment: Low Income - High Spending Customer")

    else:
        st.info("Segment: Average Customer")

# -------------------------------
# Cluster Visualization
# -------------------------------
st.subheader("Cluster Visualization")

import matplotlib.pyplot as plt

clusters = kmeans.predict(X_scaled)

plt.figure(figsize=(7,5))

for i in range(5):
    plt.scatter(
        X_scaled[clusters == i, 0],
        X_scaled[clusters == i, 1],
        label=f"Cluster {i}"
    )

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    c="black",
    s=200,
    marker="X",
    label="Centroids"
)

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segments")
plt.legend()

st.pyplot(plt)