import os
import streamlit as st
import numpy as np
import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------
# Gaussian Kernel
# -------------------------
def gaussian_kernel(x, xi, tau):
    diff = x - xi
    return np.exp(-(np.dot(diff, diff)) / (2 * tau ** 2))
import base64

def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.82)), url(\"data:image/jpg;base64,{encoded}\");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 2rem;
            border-radius: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local("house.png")
# -------------------------
# Locally Weighted Regression (with ridge regularization)
# -------------------------
def locally_weighted_regression(X_train, y_train, X_query, tau, reg_lambda=1e-2):

    m = X_train.shape[0]
    n = X_train.shape[1]
    X_train_aug = np.hstack((np.ones((m, 1)), X_train))

    I = np.eye(n + 1)
    I[0, 0] = 0  # do not regularize intercept

    y_pred = []

    for x in X_query:
        weights = np.array([gaussian_kernel(x, xi, tau) for xi in X_train])
        W = np.diag(weights)

        theta = np.linalg.pinv(
            X_train_aug.T @ W @ X_train_aug + reg_lambda * I
        ) @ X_train_aug.T @ W @ y_train

        x_aug = np.concatenate(([1], x))
        y_pred.append(x_aug @ theta)

    return np.array(y_pred)

# -------------------------
# Feature descriptions
# -------------------------

feature_descriptions = {
"CRIM": "Per capita crime rate by town",
"ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft",
"INDUS": "Proportion of non-retail business acres per town",
"CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
"NOX": "Nitric oxides concentration (parts per 10 million)",
"RM": "Average number of rooms per dwelling",
"AGE": "Proportion of owner-occupied units built prior to 1940",
"DIS": "Weighted distances to five Boston employment centres",
"RAD": "Index of accessibility to radial highways",
"TAX": "Full-value property-tax rate per $10,000",
"PTRATIO": "Pupil-teacher ratio by town",
"B": "1000(Bk - 0.63)^2 where Bk is the proportion of Black residents",
"LSTAT": "% lower status of the population"
}

# Feature list
feature_names = list(feature_descriptions.keys())

# Binary features
binary_features = ["CHAS"]

# -------------------------
# Load real Boston Housing dataset
# -------------------------

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("vikrishnan/boston-house-prices")
    file_path = os.path.join(path, "housing.csv")
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
                  'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    return df

df = load_data()

X_raw = df[feature_names].values
y_all = df['MEDV'].values

# Scale features — must use same scaler for both training and prediction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, _, y_train, _ = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)

# Keep a DataFrame of raw values for showing realistic input ranges
X = df[feature_names]

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏠 Boston Housing Price Prediction (Locally Weighted Regression)")

col1, col2 = st.columns(2)
with col1:
    tau = st.slider("Bandwidth (tau) — higher = smoother fit", 0.01, 2.0, 1.0)
with col2:
    reg_lambda = st.select_slider(
        "Regularization (λ) — higher = less overfitting",
        options=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        value=1e-2,
    )

# -------------------------
# Prediction Input Section
# -------------------------

try:

    with st.container():

        st.subheader("✍️ Enter House Features")

        # Feature descriptions panel
        with st.expander("📋 Feature Descriptions", expanded=False):
            for col in X.columns:
                if col in feature_descriptions:
                    st.write(f"- **{col}**: {feature_descriptions[col]}")

        input_data = []

        for i, feature in enumerate(X.columns):

            desc = feature_descriptions.get(feature, "")

            if feature in binary_features:

                value = st.selectbox(
                    f"{feature} (0/1)",
                    options=[0, 1],
                    format_func=lambda x: f"{x} ({'No' if x == 0 else 'Yes'})",
                    help=desc if desc else None,
                    key=f"feature_{i}",
                )

            else:

                col_min = float(X[feature].min())
                col_max = float(X[feature].max())
                col_mean = float(X[feature].mean())

                range_help = f"Range in dataset: {col_min:.4f} to {col_max:.4f}"

                value = st.number_input(
                    feature,
                    value=round(col_mean, 4),
                    help=f"{desc} | {range_help}",
                    key=f"feature_{i}",
                )

            input_data.append(value)

        # Prediction button
        if st.button("Predict House Price", key="btn_predict_house"):

            with st.spinner("⏳ Predicting house price..."):

                # Raw input → scale using the SAME scaler fitted on training data
                raw_array = np.array([input_data])
                scaled_array = scaler.transform(raw_array)

                prediction = locally_weighted_regression(
                    X_train,
                    y_train,
                    scaled_array,
                    tau=tau,
                    reg_lambda=reg_lambda,
                )[0]

                st.success(f"💰 Predicted Median House Value: ${prediction:.2f}k")

except Exception as e:
    st.error(f"Prediction error: {str(e)}")