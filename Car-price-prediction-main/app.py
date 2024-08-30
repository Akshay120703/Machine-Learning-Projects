from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('lin_reg_model.pkl', 'rb') as file:
    lin_reg_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('cars.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = int(request.form['fuel_type'])
    seller_type = int(request.form['seller_type'])
    transmission = int(request.form['transmission'])
    owner = int(request.form['owner'])

    features = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]],
                            columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])

    prediction = lin_reg_model.predict(features)
    predicted_price = round(prediction[0], 2)

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
