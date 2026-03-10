import streamlit as st
import numpy as np
import pandas as pd

# -------------------------
# Gaussian Kernel
# -------------------------
def gaussian_kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))
import base64

def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_local("house.png")
# -------------------------
# Locally Weighted Regression
# -------------------------
def locally_weighted_regression(X_train, y_train, X_query, tau):

    y_pred = []

    for x in X_query:
        weights = np.array([gaussian_kernel(x, xi, tau) for xi in X_train])
        W = np.diag(weights)

        theta = np.linalg.pinv(X_train.T @ W @ X_train) @ (X_train.T @ W @ y_train)

        y_pred.append(x @ theta)

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
# Dummy training data (replace with real dataset)
# -------------------------

np.random.seed(0)

X_train = np.random.rand(100, len(feature_names))
y_train = np.random.rand(100)

X = pd.DataFrame(X_train, columns=feature_names)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏠 Boston Housing Price Prediction (Locally Weighted Regression)")

tau = st.slider("Bandwidth (tau)", 0.01, 2.0, 0.5)

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

                range_help = f"Range in dataset: {col_min:.4f} to {col_max:.4f}"

                value = st.number_input(
                    feature,
                    value=0.0,
                    help=f"{desc} {range_help}".strip(),
                    key=f"feature_{i}",
                )

            input_data.append(value)

        # Prediction button
        if st.button("Predict House Price", key="btn_predict_house"):

            with st.spinner("⏳ Predicting house price..."):

                input_array = np.array([input_data])

                prediction = locally_weighted_regression(
                    X_train,
                    y_train,
                    input_array,
                    tau
                )[0]

                st.success(f"💰 Predicted Median House Value: {prediction:.3f}")

except Exception as e:
    st.error(f"Prediction error: {str(e)}")