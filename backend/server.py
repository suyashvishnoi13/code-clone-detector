from flask import Flask, request, jsonify
from flask_cors import CORS
from clone_detector import detect_clones

app = Flask(__name__)
CORS(app)  # Allow frontend requests

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    code1 = data.get("code1", "")
    code2 = data.get("code2", "")

    if not code1 or not code2:
        return jsonify({"error": "Missing input code"}), 400

    result = detect_clones(code1, code2)
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Clone Detector Backend is Running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
