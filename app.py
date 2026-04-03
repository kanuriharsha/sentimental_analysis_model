import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)

# Allow all origins — tighten this to your frontend domain in production
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def home():
    return "Sentiment API is running 🚀"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_api():
    # Preflight request handled automatically by flask-cors
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    result = predict(data["text"])
    return jsonify(result)


if __name__ == "__main__":
    # Railway injects PORT env variable — fall back to 10000 for local dev
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)