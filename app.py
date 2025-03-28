
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests

app = Flask(__name__)

# GitHub CSV URL
GITHUB_CSV_URL = "https://raw.githubusercontent.com/frontend-fuel/py/main/rope_ml_training_dataset.csv"

# Load dataset from GitHub
def load_data():
    try:
        response = requests.get(GITHUB_CSV_URL)
        response.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(response.text))
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Train model
def train_model(df):
    if df.empty:
        return None, 0.0

    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Status Mapping
status_mapping = {
    0: {"status": "✅ All is well", "color": "#28a745", "emoji": "🟢"},
    1: {"status": "⚠️ Medium risk", "color": "#ffc107", "emoji": "🟡"},
    2: {"status": "🚨 High Danger", "color": "#dc3545", "emoji": "🔴"},
}

# Load and train on startup
df = load_data()
model, accuracy = train_model(df)

@app.route("/")
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[f"feature_{i}"]) for i in range(1, 5)]
        prediction = model.predict([features])[0]
        response = status_mapping[prediction]
        response["prediction"] = prediction
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
