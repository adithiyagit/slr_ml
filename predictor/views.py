from django.shortcuts import render
from django.http import HttpResponse
import os
import numpy as np
import pandas as pd
import joblib  # For model saving/loading
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Get the absolute path of the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'predictor', 'food_calories.csv')

# Load dataset
df = pd.read_csv(DATASET_PATH)
X = df[['Serving Size (g)']].values  # Feature
y = df['Calories'].values  # Target variable

# Standardizing the feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# Calculate R² score
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f"Model R-squared Score: {r2:.4f}")  # Print R² score in console

# Save the model and scaler
joblib.dump(model, 'predictor/calorie_model.pkl')
joblib.dump(scaler, 'predictor/scaler.pkl')

# Index function
def index(request):
    return render(request, 'index.html')

# Prediction function
def predict_calories(request):
    if request.method == 'POST':
        serving_size = float(request.POST['serving_size'])

        # Load trained model and scaler
        model = joblib.load('predictor/calorie_model.pkl')
        scaler = joblib.load('predictor/scaler.pkl')

        # Preprocess input
        input_data = np.array([[serving_size]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_calories = model.predict(input_scaled)[0]

        return render(request, 'result.html', {'calories': predicted_calories, 'r2_score': round(r2, 4)})
    
    return render(request, 'index.html')
