from flask import Flask, render_template, request, jsonify
from joblib import load
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf


app = Flask(__name__)

# Load crop prediction model
crop_model = load("croppred.joblib")

# Load pepper model
pepper_model = tf.keras.models.load_model("my_pepper_model.h5")
pepper_class_names = ['Bacterial spot', 'Pepper bell healthy']

# Load potato model
potato_model = tf.keras.models.load_model("potatoes.h5")
potato_class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Load tomato model
tomato_model = tf.keras.models.load_model("my_tomato_model.h5")
tomato_class_names = ['Tomato_Bacterial_spot', 'Tomato_Late_blight',
                      'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_healthy']

# Pesticide recommendation dictionary
pesticide_recommendation = {
    'Bacterial spot': 'Use pesticide Copper sprays',
    'Pepper bell healthy': 'Don\'t use pesticide',
    'Early Blight': 'Use pesticide oxychloride and streptomycin solution',
    'Late Blight': 'Use pesticide oxychloride and streptomycin solution',
    'Healthy': 'Don\'t use pesticide',
    'Tomato_Bacterial_spot': 'Use Pesticide Copper-containing bactericides',
    'Tomato_Late_blight': 'Use fungicide sprays based on mandipropamid, chlorothalonil, fluazinam, mancozeb',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'There is no treatment but try to reduce whiteflys by keeping the leaves clean',
    'Tomato_healthy': 'Don\'t use pesticide'
}

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'crop_type' not in request.form:
        return jsonify({'error': 'Crop type not specified'})

    crop_type = request.form['crop_type']
    if crop_type == 'pepper':
        model = pepper_model
        class_names = pepper_class_names
    elif crop_type == 'potatoes':
        model = potato_model
        class_names = potato_class_names
    elif crop_type == 'tomatoes':
        model = tomato_model
        class_names = tomato_class_names
    else:
        return jsonify({'error': 'Invalid crop type'})

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'})

    try:
        img = Image.open(file)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        pesticide = pesticide_recommendation.get(predicted_class, 'Unknown')
        return jsonify({'predicted_class': predicted_class, 'pesticide_recommendation': pesticide})

    except Exception as e:
        return jsonify({'error': str(e)})

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
