import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset
data = pd.read_csv(r"C:\your\path\to\House Price India.csv")  # Update path accordingly

# Feature Engineering
data['total_area'] = data['living area'] + data['Area of the basement']
data['property_age'] = 2023 - data['Built Year']
data['renovated'] = (data['Renovation Year'] > 0).astype(int)

# Feature List
features = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'waterfront present',
    'grade of the house',
    'total_area',
    'property_age',
    'renovated',
    'Number ofschools nearby',
    'Distance from the airport'
]

# Define input and output
X = data[features]
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])

# Fit model
pipeline.fit(X_train, y_train)

# Save pipeline and features list together
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump((pipeline, features), f)

print("✅ Model built and saved with schools and airport distance features.")
