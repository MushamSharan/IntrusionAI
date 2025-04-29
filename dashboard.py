from flask import Flask, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
try:
    model = joblib.load('ml_training/trained_xgboost_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load a sample of the test data for simulation
try:
    test_data = pd.read_csv('data/kaggle_test_features.csv').head(5) # Load first 5 rows
except Exception as e:
    print(f"Error loading test data: {e}")
    test_data = None

@app.route('/')
def index():
    predictions = []
    predicted_labels = [] # Initialize predicted_labels
    if model is not None and test_data is not None:
        predictions = model.predict(test_data)
        print("Simulated Predictions:", predictions)
        # Map predictions back to original labels (assuming you have label_mapping)
        label_mapping = {0: 'BENIGN', 1: 'Bot', 2: 'PortScan', 3: 'DDoS', 4: 'Web Attack', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Infiltration', 13: 'Invalid Source', 14: 'Exploits'}  # Replace with your actual mapping
        predicted_labels = [label_mapping.get(p, 'UNKNOWN') for p in predictions]
    
    return render_template('index.html', predictions=predicted_labels)

if __name__ == '__main__':
    app.run(debug=True)