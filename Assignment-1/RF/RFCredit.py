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

st.title("🌳 Credit Card Fraud Detection - Random Forest")

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

    return pd.read_csv(csv_path)


# Train and cache the model
# -------------------------------
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
    if target_col:
        fraud_count = df[target_col].sum()
        st.write(f"Fraudulent Transactions: {int(fraud_count):,}")
        st.write(f"Fraud Rate: {(fraud_count/len(df)*100):.2f}%")

# Class Distribution
st.subheader("📈 Class Distribution")
if target_col:
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        fig, ax = plt.subplots(figsize=(6, 4))
        class_counts = df[target_col].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(class_counts.index, class_counts.values, color=colors, alpha=0.7)
        ax.set_xlabel("Class (0=Normal, 1=Fraud)")
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        for i, v in enumerate(class_counts.values):
            ax.text(i, v + max(class_counts.values)*0.02, str(v), ha='center')
        st.pyplot(fig)
    with col_chart2:
        st.write("**Class Breakdown:**")
        for idx, count in class_counts.items():
            label = "Normal" if idx == 0 else "Fraud"
            st.metric(label, f"{count:,}")

st.success("✅ Data loaded and preprocessed successfully!")

st.divider()

# ========================================
# 2. MODEL BUILDING
# ========================================

st.header("🤖 2. Model Building")

st.subheader("🌳 Random Forest Classifier")

# Train model once and cache it
model, scaler, X, y, X_test, y_test = train_model(df, target_col)

st.info("**Model Parameters:**\n- Algorithm: Random Forest Classifier\n- Estimators: 20\n- Max Depth: 10\n- Min Samples Split: 10\n- Class Weight: Balanced\n- SMOTE Applied: Yes")

st.success("✅ Random Forest model trained successfully!")

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

try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("🎯 Performance Metrics")
    
    cm = confusion_matrix(y_test, y_pred)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with metric_col2:
        # Calculate precision
        if cm[1, 1] + cm[0, 1] > 0:
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        else:
            precision = 0
        st.metric("Precision", f"{precision:.4f}")
    with metric_col3:
        # Calculate recall
        if cm[1, 1] + cm[1, 0] > 0:
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        else:
            recall = 0
        st.metric("Recall", f"{recall:.4f}")
    with metric_col4:
        # Calculate F1-score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        st.metric("F1-Score", f"{f1:.4f}")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("Actual Label")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col_feat1, col_feat2 = st.columns([2, 1])
    with col_feat1:
        fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
        top_features = feature_importance.head(10)
        ax_feat.barh(range(len(top_features)), top_features['Importance'], color='steelblue', alpha=0.7)
        ax_feat.set_yticks(range(len(top_features)))
        ax_feat.set_yticklabels(top_features['Feature'])
        ax_feat.set_xlabel('Importance')
        ax_feat.set_title('Top 10 Most Important Features')
        ax_feat.invert_yaxis()
        st.pyplot(fig_feat)
    with col_feat2:
        st.write("**Top 5 Features:**")
        for idx, row in feature_importance.head(5).iterrows():
            st.write(f"{row['Feature']}: {row['Importance']:.4f}")
    
except Exception as e:
    st.error(f"Model evaluation error: {str(e)}")

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
                prediction_idx = model.predict(row_scaled)[0]
                probabilities_idx = model.predict_proba(row_scaled)[0]
                fraud_probability_idx = probabilities_idx[1] if len(probabilities_idx) > 1 else probabilities_idx[0]

                if prediction_idx == 1:
                    st.error("🚨 Fraudulent Transaction Detected")
                else:
                    st.success("✓ Normal Transaction")
                st.write("Fraud Probability:", round(float(fraud_probability_idx), 4))

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

