import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("stress_data.csv")

X = data["text"]
y = data["stress"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

model.fit(X, y)

joblib.dump(model, "stress_model.pkl")

print("✅ Model trained and saved")
