import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from xgboost import XGBClassifier

# Set matplotlib font sizes
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection - XGBoost", layout="wide")
st.title("💳 Credit Card Fraud Detection using XGBoost")

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

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    return model, scaler, X, y, X_test, y_test, y_pred, y_prob, accuracy, roc


df = load_data()

# Determine target column
if "Class" in df.columns:
    target_col = "Class"
elif "fraud" in df.columns:
    target_col = "fraud"
else:
    target_col = df.columns[-1]

# Train model once and cache it
model, scaler, X, y, X_test, y_test, y_pred, y_prob, accuracy, roc = train_model(df, target_col)

# ========== SECTION 1: Dataset Overview ==========
with st.container():
    st.divider()
    st.subheader("📊 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**First 5 Rows**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Dataset Shape**")
        st.metric("Total Rows", df.shape[0])
        st.metric("Features", df.shape[1])

# ========== SECTION 2: Class Distribution ==========
with st.container():
    st.divider()
    st.subheader("📈 Class Distribution")
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.countplot(x=df[target_col], ax=ax)
    ax.set_xlabel("Class", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Class Distribution", fontsize=9, fontweight='bold')
    st.pyplot(fig)

# ========== SECTION 3: Data Preprocessing ==========
with st.container():
    st.divider()
    st.subheader("⚙️ Data Preprocessing")
    
    col_proc1, col_proc2, col_proc3, col_proc4 = st.columns(4)
    col_proc1.write("✓ Missing values removed")
    col_proc2.write("✓ Label encoding applied")
    col_proc3.write("✓ Scaled with StandardScaler")
    col_proc4.write("✓ Train-Test: 80-20")

# ========== SECTION 4: SMOTE & Model Training ==========
with st.container():
    st.divider()
    st.subheader("🤖 Model Training - XGBoost")
    
    st.write("✓ SMOTE applied to balance dataset")
    st.success("✅ XGBoost Model Trained Successfully")

# ========== SECTION 5: Model Evaluation ==========
with st.container():
    st.divider()
    st.subheader("📊 Model Evaluation")
    
    # Metrics in columns
    col_acc, col_roc = st.columns(2)
    
    with col_acc:
        st.metric("🎯 Accuracy", f"{round(accuracy, 4)}", "Test Set")
    
    with col_roc:
        st.metric("📈 ROC-AUC Score", f"{round(roc, 4)}", "Test Set")
    
    # Classification Report & Confusion Matrix in tabs
    tab1, tab2 = st.tabs(["📋 Classification Report", "🔥 Confusion Matrix"])
    
    with tab1:
        st.text(classification_report(y_test, y_pred))
    
    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, annot_kws={'fontsize': 8})
        ax2.set_xlabel("Predicted", fontsize=8)
        ax2.set_ylabel("Actual", fontsize=8)
        ax2.set_title("Confusion Matrix", fontsize=9, fontweight='bold')
        st.pyplot(fig2)

# ========== SECTION 6: Feature Importance ==========
with st.container():
    st.divider()
    st.subheader("⭐ Feature Importance")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.5))
    ax3.barh(range(len(indices)), importances[indices], color='steelblue')
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels(X.columns[indices], fontsize=8)
    ax3.set_xlabel("Importance", fontsize=8)
    ax3.set_title("Top 10 Feature Importance", fontsize=9, fontweight='bold')
    ax3.tick_params(axis='x', labelsize=7)
    st.pyplot(fig3)

# ========== SECTION 7: Manual Prediction ==========
with st.container():
    st.divider()
    st.subheader("🔮 Manual Transaction Prediction")
    
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
        predict_btn = st.button(" Predict", use_container_width=True, key="predict_btn_main")
    
    if predict_btn:
        with st.spinner("⏳ Analyzing transaction..."):
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[X.columns]  # Ensure column order matches training data
            
            try:
                input_scaled = scaler.transform(input_df)
                
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                with col_btn2:
                    if prediction == 1:
                        st.error("🚨 Fraudulent")
                    else:
                        st.success("✅ Normal")
                
                with col_btn3:
                    st.info(f"Fraud Probability: {round(float(probability), 4)}")
                
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
