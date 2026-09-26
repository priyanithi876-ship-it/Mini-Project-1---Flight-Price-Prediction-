from flask import Flask, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Model loading error:", e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Flight Price Prediction",
        "status": "API is running",
        "endpoint": "POST /predict"
    })


@app.route("/health", methods=["GET"])
def health():

    if model is None:
        return jsonify({
            "status": "error",
            "model_loaded": False
        }), 500

    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )