from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


app = Flask(__name__)

# Load your pre-trained RandomForest model
model = joblib.load('random_forest_model.pkl')

# Load the label encoder used for 'region' (if applicable)
region_encoder = joblib.load('region_encoder.pkl')  # Load the encoder you used for training

# Define the feature names (the same as when you trained the model)
features = ['year', 'month', 'day_of_week', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data and strip any extra whitespace
        year = int(request.form['year'])
        month = int(request.form['month'])
        day_of_week = int(request.form['day_of_week'])
        total_volume = float(request.form['total_volume'])
        total_bags = float(request.form['total_bags'])
        small_bags = float(request.form['small_bags'])
        large_bags = float(request.form['large_bags'])
        xlarge_bags = float(request.form['xlarge_bags'])
        region = request.form['region'].strip()  # Removes any extra whitespace

        # Example region encoding (adjust this based on your model's training)
        region_mapping = {
            'California': 0,
            'New York': 1,
            'Albany': 3, 
            'Atlanta': 4, 
            'BaltimoreWashington': 5, 
            'Boise': 6, 
            'Boston': 7,
            'BuffaloRochester': 8,
            'Charlotte': 9, 
            'Chicago': 10,
            'CincinnatiDayton': 11,
            'Columbus': 12, 
            'DallasFtWorth': 13, 
            'Denver': 14,
            'Detroit': 15, 
            'GrandRapids': 16, 
            'GreatLakes': 17, 
            'HarrisburgScranton': 18,
            'HartfordSpringfield': 19, 
            'Houston': 20, 
            'Indianapolis': 21, 
            'Jacksonville': 22,
            'LasVegas': 23, 
            'LosAngeles': 24, 
            'Louisville': 25, 
            'MiamiFtLauderdale': 26,
            'Midsouth': 27, 
            'Nashville': 28, 
            'NewOrleansMobile': 29,
            'Northeast': 30, 
            'NorthernNewEngland': 31, 
            'Orlando': 32, 
            'Philadelphia': 33,
            'PhoenixTucson': 34, 
            'Pittsburgh': 35, 
            'Plains': 36, 
            'Portland': 37,
            'RaleighGreensboro': 38, 
            'RichmondNorfolk': 39, 
            'Roanoke': 40, 
            'Sacramento': 41,
            'SanDiego': 42, 
            'SanFrancisco': 43, 
            'Seattle': 44, 
            'SouthCarolina': 45,
            'SouthCentral': 46, 
            'Southeast': 47, 
            'Spokane': 48, 
            'StLouis': 49, 
            'Syracuse': 50,
            'Tampa': 51, 
            'TotalUS': 52, 
            'West': 53, 
            'WestTexNewMexico': 54
        }
        
        # Check if region is in the mapping, or handle an unknown region
        if region not in region_mapping:
            return render_template('index.html', prediction_text="Region not recognized.")
        
        # Convert region to the numeric code expected by the model
        region_encoded = region_mapping[region]

        # Create the input feature array
        input_data = np.array([[year, month, day_of_week, total_volume, total_bags, small_bags, large_bags, xlarge_bags, region_encoded]])

        # Get model prediction
        prediction = model.predict(input_data)

        # Return the result as a string
        return render_template('index.html', prediction_text=f'Predicted Average Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
