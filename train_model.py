
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load our sample dataset
data = pd.read_csv("messages.csv")

# Check that the required columns exist
if not {"label", "message"}.issubset(data.columns):
    raise ValueError("CSV must contain label and message columns.")

data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str).str.strip()
data = data[data["label"].isin(["spam", "ham"])]
data = data[data["message"] != ""]

# We need enough examples of both classes for a stratified split
if data["label"].value_counts().min() < 2:
    raise ValueError("Add at least two examples of each label.")

# Split examples into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

# Build a text-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on held-out examples
predictions = model.predict(X_test)

print("ScamShield AI - ML Training Results")
print("-----------------------------------")
print(f"Total examples: {len(data)}")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    zero_division=0
))
import joblib

joblib.dump(model, "scamshield_model.pkl")

print("\nModel saved successfully as scamshield_model.pkl")