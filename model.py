import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("avocado.csv")
region_encoder = LabelEncoder()
data['region'] = region_encoder.fit_transform(data['region'])

# Preprocess the data (convert 'Date' to datetime and extract features)
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day_of_week'] = data['Date'].dt.dayofweek

# Prepare features and target
X = data[['year', 'month', 'day_of_week', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region']]
y = data['AveragePrice']

# Convert categorical 'region' column to numeric codes
X['region'] = X['region'].astype('category').cat.codes

# Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')

print("Model trained and saved as 'random_forest_model.pkl'")
