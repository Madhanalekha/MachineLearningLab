import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "spam.csv")
data = pd.read_csv(DATA_PATH, encoding="latin-1")

if "v1" not in data.columns or "v2" not in data.columns:
    raise ValueError("spam.csv must contain columns 'v1' (label) and 'v2' (message).")

X = data['v2'].fillna("").astype(str).str.strip()
y = data['v1'].map({"spam": 1, "ham": 0})

non_empty_mask = X != ""
X = X[non_empty_mask]
y = y[non_empty_mask]

if X.empty:
    raise ValueError("All text rows are empty after cleaning. Check the 'v2' column in spam.csv.")

# Create Pipeline (TF-IDF + SVM)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svm', LinearSVC())
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
try:
    model.fit(X_train, y_train)
except ValueError as exc:
    if "empty vocabulary" in str(exc):
        # Fallback 1: allow single-character tokens
        model = Pipeline([
            ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")),
            ('svm', LinearSVC())
        ])
        try:
            model.fit(X_train, y_train)
        except ValueError as exc2:
            if "empty vocabulary" in str(exc2):
                # Fallback 2: character n-grams for very short text
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                    ('svm', LinearSVC())
                ])
                model.fit(X_train, y_train)
            else:
                raise
    else:
        raise

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "spam_model.joblib")