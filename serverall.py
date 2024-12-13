from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load crop prediction model
crop_model = load("croppred.joblib")



yield_model = load("yield_prediction_pipeline.joblib")

# Define mapping dictionaries
market_name_map = {'Adilabad(Rythu Bazar)': 1, 'Bowenpally': 2, 'Chevella': 3,
                   'Erragadda(Rythu Bazar)': 4, 'Gajwel': 5, 'Gudimalkapur': 6,
                   'Hanmarkonda(Rythu Bazar)': 7, 'Ibrahimputnam': 8, 'Jainath': 9,
                   'Kalwakurthy': 10, 'Karimnagar': 11, 'Karimnagar(Rythu Bazar)': 12}

commodity_map = {'Tomato': 1, 'Brinjal': 2}

variety_map = {'Other': 1, 'Deshi': 2, 'Local': 3, 'Hybrid': 4, 'Arkasheela Mattigulla': 5,
               'Brinjal': 6}

month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
             'May': 5, 'June': 6, 'July': 7, 'August': 8,
             'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Load the saved Random Forest model
model_filename = 'random_forest_model.joblib'
saved_model = load(model_filename)

@app.route('/')
def index():
    return render_template('indexall.html', market_name_map=market_name_map,
                           commodity_map=commodity_map, variety_map=variety_map,
                           month_map=month_map, modal_price=None)



@app.route('/crop-predict', methods=['POST'])
def crop_predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make a prediction using the loaded model
        prediction = crop_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        return jsonify({'predicted_label': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/yield-predict', methods=['POST'])
def yield_predict():
    try:
        # Extracting data from the request form
        Area = request.form['Area']
        Item = request.form['Item']
        Year = int(request.form['Year'])
        average_rain  = float(request.form['average_rain'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])

        # Creating a pandas DataFrame from the input data
        data = {'Area': [Area], 'Item': [Item], 'Year': [Year], 'average_rain_fall_mm_per_year': [average_rain], 'pesticides_tonnes': [pesticides_tonnes], 'avg_temp': [avg_temp]}
        df = pd.DataFrame(data)

        # Make a prediction using the loaded model
        prediction = yield_model.predict(df)

        return jsonify({'predicted_label_y': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/price-predict', methods=['POST'])
def price_predict():
    # Extract user input from the form
    user_input = {
        'Market Name': int(request.form['market_name']),
        'Commodity': int(request.form['commodity']),
        'Variety': int(request.form['variety']),
        'Min Price': float(request.form['min_price']),
        'Max Price': float(request.form['max_price']),
        'Month': int(request.form['month'])
    }

    # Convert user input data to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Make predictions using the saved model
    predicted_modal_price = saved_model.predict(user_df)

    # Pass the prediction result to the result page
    return render_template('indexall.html', market_name_map=market_name_map,
                           commodity_map=commodity_map, variety_map=variety_map,
                           month_map=month_map, modal_price=predicted_modal_price[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
