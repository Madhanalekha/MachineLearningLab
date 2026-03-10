import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_drawable_canvas import st_canvas
import cv2
st.set_page_config(page_title="Character Recognition", layout="wide")
st.title("🔤 Handwritten Character Recognition (A–Z)")
st.write("Using Multilayer Perceptron (MLP)")

@st.cache_data
def load_data():
    data = pd.read_csv("A_Z Handwritten Data.csv", nrows=200000).astype("float32")
    return data
    return data

with st.spinner("🔄 Loading data..."):
    data = load_data()

st.success(f"✅ Dataset loaded: {len(data):,} samples")

st.subheader("📊 Dataset Preview")
st.write("First 10 rows of the dataset:")
st.dataframe(data.head(10), use_container_width=True)
st.write(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
st.write(f"Features: {data.shape[1] - 1} (pixel values)")
st.write(f"Label column: First column (0-25 representing A-Z)")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

letters = [chr(i) for i in range(65, 91)]
y_letters = np.array([letters[int(label)] for label in y])

st.subheader("📚 Model Training")

import os
model_exists = os.path.exists("model.pkl")

col1, col2 = st.columns([3, 1])
with col2:
    retrain = st.button("🔄 Retrain Model", type="primary")

if not model_exists or retrain:
    st.info("🔄 Training model for the first time...")
    
    with st.spinner("🔄 Splitting data..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    with st.spinner("🔄 Scaling features..."):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype('float32')
        X_test_scaled = scaler.transform(X_test).astype('float32')

    with st.spinner("🔄 Training MLP model... This may take a few minutes"):
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
            batch_size=128,
            learning_rate_init=0.001,
            alpha=0.0001
        )
        
        mlp.fit(X_train_scaled, y_train)

    with st.spinner("🔄 Evaluating model..."):
        y_train_pred = mlp.predict(X_train_scaled)
        y_test_pred = mlp.predict(X_test_scaled)
        
        # Convert numeric predictions to letters for metrics
        y_train_letters = np.array([letters[int(label)] for label in y_train])
        y_test_letters = np.array([letters[int(label)] for label in y_test])
        y_train_pred_letters = np.array([letters[int(pred)] for pred in y_train_pred])
        y_test_pred_letters = np.array([letters[int(pred)] for pred in y_test_pred])
        
        train_accuracy = accuracy_score(y_train_letters, y_train_pred_letters)
        test_accuracy = accuracy_score(y_test_letters, y_test_pred_letters)
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train_letters, y_train_pred_letters, average='weighted', zero_division=0
        )
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test_letters, y_test_pred_letters, average='weighted', zero_division=0
        )
        
        test_cm = confusion_matrix(y_test_letters, y_test_pred_letters, labels=letters)

    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': test_cm
    }
    
    # Print metrics to console
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print("\nTRAINING DATA:")
    print(f"  Accuracy:  {train_accuracy*100:.2f}%")
    print(f"  Precision: {train_precision*100:.2f}%")
    print(f"  Recall:    {train_recall*100:.2f}%")
    print(f"  F1-score:  {train_f1*100:.2f}%")
    print("\nTEST DATA:")
    print(f"  Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Precision: {test_precision*100:.2f}%")
    print(f"  Recall:    {test_recall*100:.2f}%")
    print(f"  F1-score:  {test_f1*100:.2f}%")
    print("\n" + "="*60 + "\n")
    
    joblib.dump((mlp, scaler, metrics), "model.pkl")

    st.success("✅ Model Trained Successfully!")
    st.info(f"Model saved as 'model.pkl'")
    model_exists = True



st.subheader("🔮 Upload an Image for Prediction")
st.subheader("✏️ Draw a Character for Prediction")
st.write("Draw a letter (A-Z) in the canvas below:")

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    predict_button = st.button("🔮 Predict Character", type="primary")
    clear_button = st.button("🗑️ Clear Canvas")

if predict_button and canvas_result.image_data is not None:
    try:
        loaded_data = joblib.load("model.pkl")
        if len(loaded_data) == 3:
            mlp, scaler, metrics = loaded_data
        else:
            mlp, scaler = loaded_data
            metrics = {}

        letters = [chr(i) for i in range(65, 91)]

        with st.spinner("🔄 Processing drawing..."):
            # Get the image data from canvas
            img = canvas_result.image_data.astype('uint8')
            
            # Convert RGBA to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            # Resize to 28x28
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Flatten and reshape for model
            image_array = img_resized.reshape(1, -1).astype('float32')
            
            # Scale the data
            image_array_scaled = scaler.transform(image_array)
            
            # Make prediction
            prediction_numeric = mlp.predict(image_array_scaled)
            prediction_letter = letters[int(prediction_numeric[0])]
            
            # Get prediction probabilities
            prediction_proba = mlp.predict_proba(image_array_scaled)[0]
            confidence = prediction_proba[int(prediction_numeric[0])] * 100
        
        st.success(f"### Predicted Character: **{prediction_letter}**")
        st.info(f"Confidence: {confidence:.2f}%")
        
        with st.expander("View Processed Image (28x28)"):
            st.image(img_resized, caption="Processed 28x28 Image", width=140)
    
    except FileNotFoundError:
        st.warning("⚠️ Please train the model first before making predictions!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")