from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and other necessary files
model = joblib.load('data/fraud_detection_model.pkl')
preprocessor = joblib.load('data/preprocessor.pkl')
feature_names = joblib.load('data/feature_names.pkl')

@app.route('/')
def home():
    return "Fraud Detection Model API"

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check the health of the API."""
    return jsonify(status='UP'), 200

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint to get information about the model."""
    info = {
        "model_type": str(type(model)),
        "number_of_features": len(feature_names),
        "features": feature_names
    }
    return jsonify(info), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make a prediction."""
    try:
        data = request.get_json(force=True)
        input_features = [data.get(feature) for feature in feature_names]
        input_array = np.array(input_features).reshape(1, -1)
        processed_input = preprocessor.transform(input_array)
        prediction = model.predict(processed_input)
        return jsonify({'prediction': int(prediction[0])}), 200
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    """Endpoint to get prediction probabilities."""
    try:
        data = request.get_json(force=True)
        input_features = [data.get(feature) for feature in feature_names]
        input_array = np.array(input_features).reshape(1, -1)
        processed_input = preprocessor.transform(input_array)
        probabilities = model.predict_proba(processed_input)
        return jsonify({'probabilities': probabilities.tolist()}), 200
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')