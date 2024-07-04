from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('Churn.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    features = [
        int(data['CreditScore']),
        1 if data['Geography'] == 'France' else 2 if data['Geography'] == 'Spain' else 0,
        1 if data['Gender'] == 'Male' else 0,
        int(data['Age']),
        int(data['Tenure']),
        float(data['Balance']),
        int(data['NumOfProducts']),
        int(data['HasCrCard']),
        int(data['IsActiveMember']),
        float(data['EstimatedSalary'])
    ]

    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
