from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("stress_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["message"]
    stress = model.predict([text])[0]
    return jsonify({"stress": stress})

app.run(port=5000)
