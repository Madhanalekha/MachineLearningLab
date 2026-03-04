import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")

if not os.path.exists(DATA_PATH):
    st.error("spam.csv is missing. Add spam.csv or run train_model.py.")
    st.stop()

data = pd.read_csv(DATA_PATH, encoding="latin-1")

if "v1" not in data.columns or "v2" not in data.columns:
    st.error("spam.csv must contain columns 'v1' (label) and 'v2' (message).")
    st.stop()

X = data['v2'].fillna("").astype(str).str.strip()
y = data['v1'].astype(str).str.strip().map({"spam": 1, "ham": 0})

valid_mask = (X != "") & y.notna()
X = X[valid_mask]
y = y[valid_mask].astype(int)

if X.empty:
    st.error("All text rows are empty after cleaning. Check the 'v2' column in spam.csv.")
    st.stop()

def train_svm_model(train_X, train_y):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', LinearSVC())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        train_X, train_y, test_size=0.2, random_state=42
    )

    try:
        model.fit(X_train, y_train)
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            model = Pipeline([
                ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")),
                ('svm', LinearSVC())
            ])
            try:
                model.fit(X_train, y_train)
            except ValueError as exc2:
                if "empty vocabulary" in str(exc2):
                    model = Pipeline([
                        ('tfidf', TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                        ('svm', LinearSVC())
                    ])
                    model.fit(X_train, y_train)
                else:
                    raise
        else:
            raise

    return model


model = train_svm_model(X, y)

sample_texts = X.head(500)
if not sample_texts.empty:
    sample_preds = model.predict(sample_texts)
    if len(set(sample_preds)) < 2:
        model = train_svm_model(X, y)
        #joblib.dump(model, MODEL_PATH)

# Page config
st.set_page_config(page_title="Spam Detection App", page_icon="Email")

st.title("Email Spam Detection using SVM")
st.write("Enter the email content below to check whether it is Spam or Not Spam.")

# Show a few correct predictions from the dataset
if st.button("Show Correctly Predicted Examples"):
    example_preds = model.predict(X)
    correct_mask = example_preds == y

    correct_texts = X[correct_mask]
    correct_labels = y[correct_mask]

    correct_spam = correct_texts[correct_labels == 1].head(5)
    correct_ham = correct_texts[correct_labels == 0].head(5)

    st.subheader("Correct Spam Examples")
    if correct_spam.empty:
        st.write("No correctly predicted spam examples found.")
    else:
        for msg in correct_spam:
            st.write(msg)

    st.subheader("Correct Ham Examples")
    if correct_ham.empty:
        st.write("No correctly predicted ham examples found.")
    else:
        for msg in correct_ham:
            st.write(msg)

# Text input
email_text = st.text_area("Enter Email Text Here")

# Button
if st.button("Check Email"):

    if email_text.strip() == "":
        st.warning("Please enter some email text!")
    else:
        prediction = model.predict([email_text])[0]

        if prediction == 1:
            st.error("This is a Spam Email")
        else:
            st.success("This is Not a Spam Email")