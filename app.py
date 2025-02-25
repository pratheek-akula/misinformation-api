from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))  # Dynamically set the base path



with open(os.path.join(base_path, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(os.path.join(base_path, "voting_classifier_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return "Misinformation Detection API is Running! built by pratheek"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from request
        text = data["text"]  # Extract text input

        # Transform input text using TF-IDF
        text_tfidf = tfidf_vectorizer.transform([text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]

        # Convert prediction to label
        label_mapping = {1: "True", 0: "False", 2: "Half-True", 3: "Barely-True", 4: "Pants-on-Fire"}
        predicted_label = label_mapping.get(prediction, "Unknown")

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=False)

