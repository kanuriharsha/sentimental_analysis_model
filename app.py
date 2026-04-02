from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)


@app.route("/")
def home():
    return "Sentiment API is running 🚀"


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    result = predict(data["text"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)