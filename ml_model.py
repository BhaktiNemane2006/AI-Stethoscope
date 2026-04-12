import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal)
    ]

def train_model(X, y):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X, y)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")

    joblib.dump(model, model_path)

def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")

    return joblib.load(model_path)

def predict(model, signal):
    features = extract_features(signal)
    return model.predict([features])[0]