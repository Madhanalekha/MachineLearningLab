import os

# Work around Streamlit watcher issues on some Python/Windows environments.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import base64

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Background Image
# -------------------------
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

set_bg_local(os.path.join(BASE_DIR, "heart.jpg"))

# -------------------------
# Download Dataset & Train Model (cached — runs only once)
# -------------------------
@st.cache_resource(show_spinner="Training Bayesian Network...")
def build_model():
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    data = pd.read_csv(os.path.join(path, "heart.csv"))
    return data

data = build_model()

# -------------------------
# Feature Descriptions
# -------------------------

feature_descriptions = {
    "age":"Age of the patient",
    "sex":"Gender (1 = Male, 0 = Female)",
    "cp":"Chest pain type",
    "trestbps":"Resting blood pressure",
    "chol":"Serum cholesterol level",
    "fbs":"Fasting blood sugar > 120 mg/dl",
    "restecg":"Resting ECG results",
    "thalach":"Maximum heart rate achieved",
    "exang":"Exercise induced angina",
    "oldpeak":"ST depression induced by exercise",
    "slope":"Slope of peak exercise ST segment",
    "ca":"Number of major vessels colored by fluoroscopy",
    "thal":"Thalassemia"
}

feature_names = list(feature_descriptions.keys())
binary_features = ["sex","fbs","exang"]
numeric_features = [f for f in feature_names if f not in binary_features]

# -------------------------
# Train Bayesian Network (cached)
# -------------------------
@st.cache_resource(show_spinner="Training Bayesian Network...")
def train_model(_data):
    df = _data.copy()

    model = DiscreteBayesianNetwork([
        ('age','target'), ('sex','target'), ('cp','target'),
        ('trestbps','target'), ('chol','target'), ('fbs','target'),
        ('restecg','target'), ('thalach','target'), ('exang','target'),
        ('oldpeak','target'), ('slope','target'), ('ca','target'),
        ('thal','target')
    ])

    # Discretize continuous features; store bin edges for raw-value mapping
    bin_labels = {}   # {feature: [bin_string, ...]}
    bin_edges  = {}   # {feature: np.array of edges} — used to map raw inputs
    binned_features = set()
    feat_min = {}     # raw min/max for UI hints
    feat_max = {}

    for f in numeric_features:
        feat_min[f] = float(_data[f].min())
        feat_max[f] = float(_data[f].max())

        if df[f].nunique() <= 8:
            df[f] = df[f].astype(int)
            bin_labels[f] = sorted(df[f].unique().tolist())
            continue

        binned_features.add(f)
        try:
            _, edges = pd.qcut(df[f], q=4, retbins=True, duplicates="drop")
            if len(edges) < 2:
                raise ValueError()
            binned = pd.cut(df[f], bins=edges, include_lowest=True)
        except Exception:
            binned = pd.cut(df[f], bins=4, include_lowest=True)
            edges = binned.cat.categories
            edges = np.array([iv.left for iv in edges] + [edges[-1].right], dtype=float)

        bin_edges[f]  = np.array(edges, dtype=float)
        df[f]         = binned.astype(str)
        bin_labels[f] = [str(c) for c in binned.cat.categories]

    model.fit(df, estimator=MaximumLikelihoodEstimator)
    inference = VariableElimination(model)

    return inference, bin_labels, bin_edges, binned_features, feat_min, feat_max

inference, bin_labels, bin_edges, binned_features, feat_min, feat_max = train_model(data)

def raw_to_bin(feature, raw_value):
    """Map a raw numeric value to the correct bin label string."""
    edges = bin_edges[feature]
    # First bin is inclusive on both sides (include_lowest=True)
    if raw_value <= edges[0]:
        idx = 0
    elif raw_value > edges[-1]:
        idx = len(bin_labels[feature]) - 1
    else:
        idx = int(np.searchsorted(edges[1:], raw_value, side='left'))
        idx = min(idx, len(bin_labels[feature]) - 1)
    return bin_labels[feature][idx]

# -------------------------
# Streamlit UI
# -------------------------

st.title("Heart Disease Prediction (Bayesian Network)")

st.subheader("Enter Patient Medical Details")

# Feature description panel
with st.expander("Feature Descriptions"):

    for f in feature_names:

        st.write(f"**{f}** : {feature_descriptions[f]}")

# -------------------------
# Input Section
# -------------------------

input_data = []

for feature in feature_names:

    desc = feature_descriptions.get(feature, "")

    if feature in binary_features:

        value = st.selectbox(
            f"{feature} (0/1)",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help=desc,
            key=f"inp_{feature}"
        )
        input_data.append(value)

    elif feature in binned_features:

        # Accept raw value — convert to bin at prediction time
        raw_val = st.number_input(
            feature,
            min_value=float(feat_min[feature]),
            max_value=float(feat_max[feature]),
            value=float(round((feat_min[feature] + feat_max[feature]) / 2, 2)),
            help=f"{desc} | Range: {feat_min[feature]:.1f} – {feat_max[feature]:.1f}",
            key=f"inp_{feature}"
        )
        input_data.append((feature, raw_val, 'raw'))

    else:

        value = st.selectbox(
            feature,
            options=bin_labels[feature],
            help=f"{desc} | Categorical",
            key=f"inp_{feature}"
        )
        input_data.append(value)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Heart Disease"):

    with st.spinner("Analyzing patient data..."):

        # Build evidence dict — convert raw continuous values to their bin labels
        patient = {}
        raw_iter = iter(input_data)
        for feature in feature_names:
            val = next(raw_iter)
            if isinstance(val, tuple) and val[2] == 'raw':
                patient[val[0]] = raw_to_bin(val[0], val[1])
            else:
                patient[feature] = val

        result = inference.query(
            variables=['target'],
            evidence=patient
        )

        prob = result.values[1]

        if prob > 0.5:

            st.error(f"High Risk of Heart Disease (Probability: {prob:.2f})")

        else:

            st.success(f"Low Risk of Heart Disease (Probability: {prob:.2f})")
