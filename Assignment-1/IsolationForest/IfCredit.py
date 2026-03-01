import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

st.title("🔍 Credit Card Fraud Detection - Isolation Forest")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "card_transdata.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at: {csv_path}")
        st.stop()
    
    return pd.read_csv(csv_path)


# Train and cache the model
# =========================
@st.cache_resource
def train_model(df, contamination_value):
    # Determine target column
    if "Class" in df.columns:
        target_col = "Class"
    elif "fraud" in df.columns:
        target_col = "fraud"
    else:
        target_col = df.columns[-1]
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination_value,
        random_state=42
    )
    
    iso_model.fit(X_train_scaled)
    
    y_pred = iso_model.predict(X_test_scaled)
    # Convert -1 to 1 (Fraud), 1 to 0 (Normal)
    y_pred = np.where(y_pred == -1, 1, 0)
    
    anomaly_scores = iso_model.decision_function(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, -anomaly_scores)
    
    return iso_model, scaler, X, y, X_test, y_test, y_pred, anomaly_scores, accuracy, roc_score

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head())

# =========================
# Data Preparation
# =========================
st.write("## ⚙️ Data Preparation")

st.success("Data scaled successfully!")

# =========================
# Train Isolation Forest
# =========================
st.write("## 🚀 Train Isolation Forest Model")

contamination_value = st.slider(
    "Select Contamination (Fraud Percentage)",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001
)

# Train model with cached function
iso_model, scaler, X, y, X_test, y_test, y_pred, anomaly_scores, accuracy, roc_score = train_model(df, contamination_value)

st.success("Model trained successfully!")

# =========================
# Evaluation
# =========================
st.write("## 📊 Model Evaluation")

st.write("### Accuracy")
st.write(round(accuracy, 4))

# Classification Report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================
st.write("### Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# =========================
# ROC Curve
# =========================
st.write("### ROC Curve")

fpr, tpr, _ = roc_curve(y_test, -anomaly_scores)

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.plot(fpr, tpr)
ax2.plot([0, 1], [0, 1])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
st.pyplot(fig2)

st.write("### ROC-AUC Score")
st.write(round(roc_score, 4))

# =========================
# Manual Prediction Section
# =========================
st.write("## 🔮 Predict for New Transaction")

st.write("Enter transaction details to predict if it's fraudulent:")

# Feature descriptions
feature_descriptions = {
    "distance_from_home": "The distance from home where the transaction happened.",
    "distance_from_last_transaction": "The distance from last transaction happened.",
    "ratio_to_median_purchase_price": "Ratio of purchased price transaction to median purchase price.",
    "repeat_retailer": "Is the transaction happened from same retailer.",
    "used_chip": "Is the transaction through chip (credit card).",
    "used_pin_number": "Is the transaction happened by using PIN number.",
    "online_order": "Is the transaction an online order.",
}

# Identify binary features
binary_features = []
for feature in X.columns:
    unique_vals = X[feature].unique()
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        binary_features.append(feature)

with st.expander("📋 Feature Descriptions", expanded=False):
    for feature, desc in feature_descriptions.items():
        if feature in X.columns:
            st.write(f"**{feature}**: {desc}")

# Create columns for input fields - 3 inputs per row
input_data = {}
num_features = len(X.columns)
cols_per_row = 3

st.write("**Enter values below:**")

for i in range(0, num_features, cols_per_row):
    cols = st.columns(min(cols_per_row, num_features - i))
    for j, col in enumerate(cols):
        idx = i + j
        if idx < num_features:
            feature = X.columns[idx]
            with col:
                desc = feature_descriptions.get(feature, "")
                
                if feature in binary_features:
                    # Use selectbox for binary features
                    display_label = f"{feature} (0/1)"
                    value = st.selectbox(
                        display_label,
                        options=[0, 1],
                        format_func=lambda x: f"{x} ({'No' if x == 0 else 'Yes'})",
                        help=desc if desc else None,
                        key=f"feature_{idx}",
                    )
                    input_data[feature] = value
                else:
                    # Use number input for continuous features
                    is_distance_feature = "distance" in feature.lower()
                    display_label = f"{feature} (km)" if is_distance_feature else feature
                    
                    # Get min/max from training data for context
                    col_min = float(X[feature].min())
                    col_max = float(X[feature].max())
                    range_help = f"Range: {col_min:.2f} to {col_max:.2f}"
                    
                    value = st.number_input(
                        display_label,
                        value=0.0,
                        help=f"{desc} {range_help}".strip(),
                        key=f"feature_{idx}",
                        format="%.4f"
                    )
                    input_data[feature] = value

# Prediction button and results
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    predict_btn = st.button("🚀 Predict", use_container_width=True, key="predict_btn_main")

if predict_btn:
    with st.spinner("⏳ Analyzing transaction..."):
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[X.columns]  # Ensure column order matches training data
        
        try:
            input_scaled = scaler.transform(input_df)
            
            prediction = iso_model.predict(input_scaled)[0]
            # Convert -1 to 1 (Fraud), 1 to 0 (Normal)
            prediction = 1 if prediction == -1 else 0
            
            anomaly_score = iso_model.decision_function(input_scaled)[0]
            fraud_probability = 1 / (1 + np.exp(anomaly_score))  # Convert to probability
            
            with col_btn2:
                if prediction == 1:
                    st.error("🚨 Fraudulent")
                else:
                    st.success("✅ Normal")
            
            with col_btn3:
                st.info(f"Fraud Probability: {round(float(fraud_probability), 4)}")
            
            # Display entered values summary
            with st.expander("📊 Transaction Details Summary", expanded=True):
                summary_cols = st.columns(3)
                for idx, (feature, value) in enumerate(input_data.items()):
                    col_idx = idx % 3
                    with summary_cols[col_idx]:
                        is_distance_feature = "distance" in feature.lower()
                        unit = " (km)" if is_distance_feature else ""
                        st.metric(f"{feature}{unit}", f"{value:.2f}")
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")