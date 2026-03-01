import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")


# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "card_transdata.csv")

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at: {csv_path}")
        st.stop()

    return pd.read_csv(csv_path)


# Train and cache the model
# -------------------------------
@st.cache_resource
def train_model(df, target_col):
    df = df.dropna()

    if df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        k_neighbors = min(5, len(X_train) - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        if X_train_smote.shape[0] > 100000:
            sample_indices = np.random.choice(X_train_smote.shape[0], size=100000, replace=False)
            X_train_smote = X_train_smote[sample_indices]
            if hasattr(y_train_smote, "values"):
                y_train_smote = y_train_smote.values[sample_indices]
            else:
                y_train_smote = np.array(y_train_smote)[sample_indices]
    except Exception as e:
        X_train_smote, y_train_smote = X_train, y_train

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_smote, y_train_smote)

    return model, scaler, X, y, X_test, y_test


df = load_data()

# Train model once and cache it
model, scaler, X, y, X_test, y_test = train_model(df, "Class" if "Class" in df.columns else ("fraud" if "fraud" in df.columns else df.columns[-1]))

# Determine target column
target_col = None
if "Class" in df.columns:
    target_col = "Class"
elif "fraud" in df.columns:
    target_col = "fraud"
elif len(df.columns) > 0:
    target_col = df.columns[-1]
else:
    st.error("No valid target column found in dataset")
    st.stop()

# Get binary features
binary_features = []
for col in X.columns:
    unique_vals = pd.Series(X[col]).dropna().unique()
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
        binary_features.append(col)

feature_descriptions = {
    "distance_from_home": "The distance from home where the transaction happened.",
    "distance_from_last_transaction": "The distance from last transaction happened.",
    "ratio_to_median_purchase_price": "Ratio of purchased price transaction to median purchase price.",
    "repeat_retailer": "Is the transaction happened from same retailer.",
    "used_chip": "Is the transaction through chip (credit card).",
    "used_pin_number": "Is the transaction happened by using PIN number.",
    "online_order": "Is the transaction an online order.",
}

# Display Dataset Info
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("### First 5 Rows")
    st.dataframe(df.head())
with col2:
    st.write("### Dataset Shape")
    st.write(df.shape)

# Class Distribution
st.subheader("Class Distribution")
if target_col:
    fig, ax = plt.subplots()
    sns.countplot(x=df[target_col], ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Model Evaluation
st.subheader("Model Evaluation")
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Accuracy")
    st.write(round(accuracy, 4))
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)
except Exception as e:
    st.error(f"Model evaluation error: {str(e)}")

# -------------------------------
# Manual Prediction Section
# -------------------------------
st.subheader("Manual Transaction Prediction")

try:
    with st.container():
        st.subheader("Predict Using Existing Transaction Index")
        st.caption(f"Valid index range: 0 to {len(X) - 1}")

        index_value = st.number_input(
            "Enter transaction index",
            min_value=0,
            max_value=max(len(X) - 1, 0),
            value=0,
            step=1,
            key="index_input",
        )

        if st.button("Predict by Index", key="btn_predict_index"):
            with st.spinner("⏳ Making prediction..."):
                idx = int(index_value)
                row_values = X.iloc[[idx]]
                row_scaled = scaler.transform(row_values)
                prediction_idx = model.predict(row_scaled)[0]
                probabilities_idx = model.predict_proba(row_scaled)[0]
                fraud_probability_idx = probabilities_idx[1] if len(probabilities_idx) > 1 else probabilities_idx[0]

                if prediction_idx == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✓ Normal Transaction")
                st.write("Fraud Probability:", round(float(fraud_probability_idx), 4))

    with st.container():
        st.subheader("Predict for New Transaction")
        with st.expander("Feature descriptions", expanded=False):
            for col in X.columns:
                if col in feature_descriptions:
                    st.write(f"- **{col}**: {feature_descriptions[col]}")

        input_data = []
        for i, feature in enumerate(X.columns):
            is_distance_feature = "distance" in feature.lower()
            display_label = f"{feature} (km)" if is_distance_feature else feature
            desc = feature_descriptions.get(feature, "")

            if feature in binary_features:
                value = st.selectbox(
                    f"{display_label} (0/1)",
                    options=[0, 1],
                    format_func=lambda x: f"{x} ({'No' if x == 0 else 'Yes'})",
                    help=desc if desc else None,
                    key=f"feature_{i}",
                )
            else:
                col_min = float(df[feature].min())
                col_max = float(df[feature].max())
                range_help = f"Range in dataset: {col_min:.4f} to {col_max:.4f}"
                value = st.number_input(
                    display_label,
                    value=0.0,
                    help=f"{desc} {range_help}".strip(),
                    key=f"feature_{i}",
                )
            input_data.append(value)

        if st.button("Predict for New Transaction", key="btn_predict_manual"):
            with st.spinner("⏳ Analyzing transaction..."):
                input_df = pd.DataFrame([input_data], columns=X.columns)
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]

                if prediction == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✓ Normal Transaction")
                st.write("Fraud Probability:", round(float(fraud_probability), 4))
except Exception as e:
    st.error(f"Prediction error: {str(e)}")

