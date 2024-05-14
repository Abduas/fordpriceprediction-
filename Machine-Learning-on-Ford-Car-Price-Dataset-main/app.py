#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the XGBoost model
model = XGBRegressor()
model.load_model("xgb_model.json")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the data from the form
        year = int(request.form['year'])
        transmission = int(request.form['transmission'])
        mileage = int(request.form['mileage'])
        fuel_type = int(request.form['fuelType'])
        tax = int(request.form['tax'])
        mpg = float(request.form['mpg'])
        engine_size = float(request.form['engineSize'])

        # Create a DataFrame with the new data
        new_data = {'year': year, 'transmission': transmission, 'mileage': mileage,
                    'fuelType': fuel_type, 'tax': tax, 'mpg': mpg, 'engineSize': engine_size}
        new_df = pd.DataFrame(new_data, index=[0])

        # Apply the same preprocessing as in your original code
        scaler = StandardScaler()
        new_df_scaled = scaler.fit_transform(new_df)

        # Predict the price
        predicted_price = model.predict(new_df_scaled)

        return render_template('result.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
