# High-Tech-E-Mail-Detector-

# pip install scikit-learn pandas joblib
import re, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def url_flag(text): 
    return 1 if re.search(r'https?://', text) else 0

# Example tiny dataset (replace with your CSV)
data = pd.DataFrame([
    {"text": "Verify your account urgently: http://bad.link", "label": 1},
    {"text": "Team meeting at 3 PM. Agenda attached.", "label": 0},
    {"text": "Your package is on hold. Update payment now.", "label": 1},
    {"text": "Invoice for last month attached.", "label": 0},
])

# Simple TF-IDF + LogisticRegression pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.3, random_state=42, stratify=data["label"]
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(classification_report(y_test, pred, digits=3))

joblib.dump(pipe, "phishing_model.joblib")