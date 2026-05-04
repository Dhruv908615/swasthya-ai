import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def train_model():
    df = pd.read_csv("data/symptoms_data.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(df["symptoms"], df["risk"])

    joblib.dump(model, "health_risk_model.pkl")

    print("✅ Model trained and saved as health_risk_model.pkl")


if __name__ == "__main__":
    train_model()