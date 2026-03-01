"""
Semiconductor Wafer Defect Classification using Support Vector Machine (SVM)

This Streamlit application implements a complete ML pipeline for classifying
semiconductor wafer defects using SVM with RBF kernel. It includes data preprocessing,
model training, evaluation, and prediction capabilities.

Author: ML Lab
Date: 2026
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


def _running_in_streamlit() -> bool:
    """Check if script is running in Streamlit context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if not _running_in_streamlit():
    script_path = os.path.abspath(__file__).replace("\\", "/")
    print("This app must be run with Streamlit:")
    print(f"  streamlit run \"{script_path}\"")
    raise SystemExit(0)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Wafer Defect Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ============================================================================
# SIDEBAR - DATASET SELECTION
# ============================================================================
st.sidebar.title("📊 Configuration")
st.sidebar.markdown("---")

default_csv_path = os.path.join(
    os.path.dirname(__file__), "semiconductor_wafer_defect_dataset.csv"
)

use_default = st.sidebar.checkbox(
    "Use Default Dataset",
    value=os.path.exists(default_csv_path),
    help="Load the semiconductor wafer defect dataset from the local folder",
)

# ============================================================================
# MAIN TITLE AND DESCRIPTION
# ============================================================================
st.title("🔬 Semiconductor Wafer Defect Classification")
st.markdown(
    """
    **Support Vector Machine (SVM) Based Classification**
    
    This application demonstrates a complete machine learning pipeline for classifying
    semiconductor wafer defects using SVM with RBF kernel.
    """
)

# ============================================================================
# LOAD DATASET
# ============================================================================
uploaded_file = None
df = None

if use_default and os.path.exists(default_csv_path):
    try:
        df = pd.read_csv(default_csv_path)
        st.sidebar.success("✓ Default dataset loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {str(e)}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Wafer Training Dataset (CSV)", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✓ Dataset uploaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {str(e)}")
            st.stop()

if df is None:
    st.info("👈 Please select or upload a dataset using the sidebar to continue.")
    st.stop()

# ============================================================================
# DATA EXPLORATION & PREPROCESSING
# ============================================================================
st.markdown("---")
st.header("📈 Step 1: Data Exploration")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dataset Shape")
    st.metric(label="Number of Samples", value=df.shape[0])
    st.metric(label="Number of Features", value=df.shape[1])

with col2:
    st.subheader("Dataset Info")
    st.info(f"✓ Dataset columns: {', '.join(df.columns.tolist())}")

# Display dataset preview
st.subheader("Dataset Preview (First 5 Rows)")
st.dataframe(df.head(), use_container_width=True)

# Check for required column
if "defect_label" not in df.columns:
    st.error(
        "❌ Dataset must contain a 'defect_label' column for classification."
    )
    st.stop()

# Prepare data
df = df.copy()
if "wafer_id" in df.columns:
    df = df.drop(columns=["wafer_id"])

X = df.drop("defect_label", axis=1)
y = df["defect_label"].astype(int)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

st.subheader("Feature Information")
st.write(
    f"**Numeric Features:** {len(numeric_features)} - {', '.join(numeric_features)}"
)
if categorical_features:
    st.write(
        f"**Categorical Features:** {len(categorical_features)} - {', '.join(categorical_features)}"
    )
else:
    st.write("**Categorical Features:** None")

# ============================================================================
# CLASS DISTRIBUTION VISUALIZATION
# ============================================================================
st.header("📊 Step 2: Class Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts = y.value_counts().sort_index()
    class_labels = ["No Defect (0)", "Defect (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(class_labels, class_counts.values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    st.pyplot(fig)

with col2:
    st.metric("No Defect Samples", class_counts.get(0, 0))
    st.metric("Defect Samples", class_counts.get(1, 0))
    defect_percentage = (class_counts.get(1, 0) / len(y)) * 100
    st.metric("Defect %", f"{defect_percentage:.2f}%")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
st.header("⚙️ Step 3: Data Preprocessing")

with st.expander("Preprocessing Details", expanded=True):
    st.markdown(
        """
        **Preprocessing Pipeline:**
        
        1. **Numeric Features:**
           - Handle missing values with mean imputation
           - Standardize features using StandardScaler
        
        2. **Categorical Features:**
           - Handle missing values with most frequent strategy
           - One-hot encode categorical variables
        
        3. **Train-Test Split:**
           - 80% training, 20% testing
           - Stratified split to maintain class distribution
        """
    )

# Create preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.success(
    f"✓ Data split complete - Train: {len(X_train)} samples, Test: {len(X_test)} samples"
)

# ============================================================================
# MODEL TRAINING
# ============================================================================
st.header("🤖 Step 4: SVM Model Training")

with st.spinner("Training SVM model with RBF kernel..."):
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("svm", SVC(kernel="rbf", random_state=42, probability=True)),
        ]
    )
    model.fit(X_train, y_train)

st.success("✓ Model training completed successfully!")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
st.header("📋 Step 5: Model Evaluation")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
with col2:
    st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
with col3:
    st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
with col4:
    st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")

# Classification Report
st.subheader("Classification Report")
report_text = classification_report(
    y_test, y_pred, target_names=["No Defect", "Defect"]
)
st.text(report_text)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

col1, col2 = st.columns([1.5, 1])
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=["No Defect", "Defect"],
        yticklabels=["No Defect", "Defect"],
        ax=ax,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    st.pyplot(fig)

with col2:
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"""
        **Confusion Matrix Breakdown:**
        
        - **True Negatives (TN):** {tn}
        - **False Positives (FP):** {fp}
        - **False Negatives (FN):** {fn}
        - **True Positives (TP):** {tp}
        """
    )

# ============================================================================
# MANUAL PREDICTION
# ============================================================================
st.markdown("---")
st.header("🔮 Step 6: Manual Prediction")

st.subheader("Enter Feature Values for Single Wafer Prediction")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    for i, col in enumerate(numeric_features):
        if i % 2 == 0:
            input_data[col] = st.number_input(
                f"📊 {col}",
                value=float(X[col].mean()),
                format="%.4f",
                help=f"Range: {X[col].min():.2f} - {X[col].max():.2f}",
            )

with col2:
    for i, col in enumerate(numeric_features):
        if i % 2 == 1:
            input_data[col] = st.number_input(
                f"📊 {col}",
                value=float(X[col].mean()),
                format="%.4f",
                help=f"Range: {X[col].min():.2f} - {X[col].max():.2f}",
            )

for col in categorical_features:
    options = sorted(X[col].dropna().unique().tolist())
    input_data[col] = st.selectbox(f"📁 {col}", options)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🔍 Predict Defect", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        predicted_label = "Defect ⚠️" if int(prediction) == 1 else "No Defect ✓"
        confidence = (
            prediction_proba[1] if int(prediction) == 1 else prediction_proba[0]
        )

        col_result, col_conf = st.columns(2)
        with col_result:
            if int(prediction) == 1:
                st.error(f"**Prediction: {predicted_label}**")
            else:
                st.success(f"**Prediction: {predicted_label}**")

        with col_conf:
            st.info(f"**Confidence: {confidence*100:.2f}%**")

# ============================================================================
# BATCH PREDICTION
# ============================================================================
st.markdown("---")
st.header("📦 Step 7: Batch Prediction")

st.subheader("Upload Multiple Wafers for Bulk Prediction")

new_file = st.file_uploader(
    "Upload CSV file (without defect_label column)",
    type=["csv"],
    key="batch_prediction",
)

if new_file is not None:
    try:
        new_df = pd.read_csv(new_file)

        st.subheader("Input Dataset Preview")
        st.dataframe(new_df.head(), use_container_width=True)

        if "wafer_id" in new_df.columns:
            new_df_processed = new_df.drop(columns=["wafer_id"])
        else:
            new_df_processed = new_df.copy()

        new_predictions = model.predict(new_df_processed)
        new_predictions_proba = model.predict_proba(new_df_processed)

        new_df["Predicted_Defect"] = [
            "Defect" if int(p) == 1 else "No Defect" for p in new_predictions
        ]
        new_df["Confidence_%"] = [
            (p[1] if int(new_predictions[i]) == 1 else p[0]) * 100
            for i, p in enumerate(new_predictions_proba)
        ]

        st.subheader("Prediction Results")
        st.dataframe(new_df, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            defect_count = (new_df["Predicted_Defect"] == "Defect").sum()
            st.metric("Defective Wafers", defect_count)

        with col2:
            no_defect_count = (new_df["Predicted_Defect"] == "No Defect").sum()
            st.metric("Non-Defective Wafers", no_defect_count)

        with col3:
            avg_confidence = new_df["Confidence_%"].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}%")

        # Download results
        csv = new_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="wafer_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><small>🔬 Semiconductor Wafer Defect Classification | SVM with RBF Kernel</small></p>
        <p><small>Machine Learning Laboratory | 2026</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
