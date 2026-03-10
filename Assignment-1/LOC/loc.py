import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection - LOF Model")

# ========================================
# 1. DATA PREPROCESSING
# ========================================

st.header("📂 1. Data Preprocessing")

# Load Dataset
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "card_transdata.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at: {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    
    # Limit dataset size for faster processing
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
    
    return df


# Train and cache the model
# =========================
@st.cache_resource
def train_model(_df, target_col):
    _df = _df.dropna()
    
    if _df[target_col].dtype == "object":
        le = LabelEncoder()
        _df[target_col] = le.fit_transform(_df[target_col])
    
    X = _df.drop(target_col, axis=1)
    y = _df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pure Unsupervised LOF - Train on features only, no SMOTE
    # Labels are only used for evaluation purposes
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        novelty=True,
        n_jobs=-1
    )
    
    # Fit LOF on training features only (unsupervised)
    lof_model.fit(X_train)
    
    # Predict on test set
    y_pred = lof_model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)
    
    # Calculate metrics using labels for evaluation only
    accuracy = accuracy_score(y_test, y_pred)
    
    lof_scores = lof_model.decision_function(X_test)
    roc = roc_auc_score(y_test, -lof_scores)
    
    return lof_model, scaler, X, y, X_test, y_test, y_pred, lof_scores, accuracy, roc

df = load_data()

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

# Train model once and cache it
lof_model, scaler, X, y, X_test, y_test, y_pred, lof_scores, accuracy, roc = train_model(df, target_col)

# Display Dataset Info
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)
with col1:
    st.write("**Dataset Preview (First 5 Rows)**")
    st.dataframe(df.head())
with col2:
    st.write("**Dataset Information**")
    st.write(f"Shape: {df.shape}")
    st.write(f"Total Records: {len(df):,}")
    st.write(f"Total Features: {len(df.columns)}")
   
st.success("✅ Data loaded and preprocessed successfully!")

st.divider()

# ========================================
# 2. MODEL BUILDING
# ========================================

st.header("🤖 2. Model Building")

st.subheader("🚀 Local Outlier Factor (LOF) Configuration")

st.info("**Model Parameters:**\n- Algorithm: Local Outlier Factor\n- Neighbors: 20\n- Contamination: 0.01\n- Mode: Novelty Detection")

st.success("✅ LOF Model trained successfully!")

st.write(f"**Training Details:**")
metric_train_col1, metric_train_col2, metric_train_col3 = st.columns(3)
with metric_train_col1:
    st.metric("Training Samples", f"{len(X_test)*4:,}")
with metric_train_col2:
    st.metric("Test Samples", f"{len(X_test):,}")
with metric_train_col3:
    st.metric("Features Used", len(X.columns))

st.divider()

# ========================================
# 3. MODEL EVALUATION
# ========================================

st.header("📊 3. Model Evaluation")

st.subheader("🎯 Performance Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Accuracy", f"{accuracy:.4f}")
with metric_col2:
    st.metric("ROC-AUC Score", f"{roc:.4f}")
with metric_col3:
    # Calculate precision from confusion matrix
    cm_temp = confusion_matrix(y_test, y_pred)
    if cm_temp[1, 1] + cm_temp[0, 1] > 0:
        precision = cm_temp[1, 1] / (cm_temp[1, 1] + cm_temp[0, 1])
    else:
        precision = 0
    st.metric("Precision", f"{precision:.4f}")

# Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🔢 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, -lof_scores)

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC Curve (AUC = {round(roc, 4)})')
ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)
st.pyplot(fig_roc)

st.divider()

# ========================================
# 4. PREDICTION
# ========================================

st.header("🔮 4. Prediction for Transactions")

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

try:
    with st.container():
        st.subheader("🔍 Predict Using Existing Transaction Index")
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
                prediction_idx = lof_model.predict(row_scaled)[0]
                prediction_idx = 1 if prediction_idx == -1 else 0
                risk_score_idx = -lof_model.decision_function(row_scaled)[0]

                if prediction_idx == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✓ Normal Transaction")
                st.write("Risk Score:", round(float(risk_score_idx), 4))

    with st.container():
        st.subheader("✍️ Predict for New Transaction")
        with st.expander("📋 Feature Descriptions", expanded=False):
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
                col_min = float(X[feature].min())
                col_max = float(X[feature].max())
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
                prediction = lof_model.predict(input_scaled)[0]
                prediction = 1 if prediction == -1 else 0
                risk_score = -lof_model.decision_function(input_scaled)[0]

                if prediction == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✓ Normal Transaction")
                st.write("Risk Score:", round(float(risk_score), 4))
except Exception as e:
    st.error(f"Prediction error: {str(e)}")