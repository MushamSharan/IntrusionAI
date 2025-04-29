import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler  # Import StandardScaler (though joblib handles it)

st.title("AI-NIDS Real-time Detection Dashboard")

# Load the trained XGBoost model
try:
    model = joblib.load('ml_training/trained_xgboost_model.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load the feature names
try:
    feature_names = pd.read_csv('data/kaggle_train_features.csv', nrows=1).columns.tolist()
    numerical_features = [col for col in feature_names if col not in ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']] # Identify numerical features
except Exception as e:
    st.error(f"Error loading feature names: {e}")
    feature_names = None
    numerical_features = None

# Load the scaler
try:
    scaler = joblib.load('data/scaler.joblib')
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    scaler = None

# Load the label mapping
label_mapping = {0: 'BENIGN', 1: 'Bot', 2: 'PortScan', 3: 'DDoS', 4: 'Web Attack', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Infiltration', 13: 'Invalid Source', 14: 'Exploits'}

st.sidebar.header("Enter Network Traffic Features")
input_features = {}
if feature_names:
    for feature in feature_names:
        default_value = 0.0
        if feature == 'Flow ID':
            default_value = ''
        elif feature in ['Src IP', 'Dst IP']:
            default_value = ''
        elif feature in ['Src Port', 'Dst Port']:
            default_value = 80
        input_features[feature] = st.sidebar.text_input(feature, value=default_value)
else:
    st.sidebar.warning("Feature names not loaded.")

if st.sidebar.button("Predict"):
    if model and feature_names and scaler and all(input_features.values()):
        try:
            input_df = pd.DataFrame([input_features])
            numerical_input = input_df[numerical_features].astype(float) # Extract and convert numerical features
            scaled_input = scaler.transform(numerical_input)

            # Create a DataFrame with scaled numerical features and other (non-scaled) features
            scaled_input_df = pd.DataFrame(scaled_input, columns=numerical_features)
            final_input = pd.concat([scaled_input_df, input_df.drop(columns=numerical_features)], axis=1)
            final_input = final_input[feature_names] # Ensure correct column order

            prediction_probabilities = model.predict_proba(final_input)[0]
            predicted_numerical = np.argmax(prediction_probabilities)
            predicted_label = label_mapping.get(predicted_numerical, 'UNKNOWN')

            st.subheader("Prediction:")
            st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>Predicted Traffic Type: <span style='color: green;'>{predicted_label}</span></p>", unsafe_allow_html=True)
            st.write(f"Class ID: {predicted_numerical}")

            prob_df = pd.DataFrame({'Class': list(label_mapping.values()), 'Probability': prediction_probabilities})
            fig_prob = px.bar(prob_df, x='Class', y='Probability', title='Prediction Probabilities')
            st.plotly_chart(fig_prob)

        except ValueError as ve:
            st.error(f"Please enter valid numerical values for the numerical features: {ve}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    elif not all(input_features.values()):
        st.sidebar.warning("Please fill in all the input features.")
    elif not scaler:
        st.warning("Scaler not loaded. Ensure 'data/scaler.joblib' exists.")
    else:
        st.warning("Model or feature names not loaded.")

st.markdown("---")
st.info("Enter network flow characteristics in the sidebar and click 'Predict' to see the model's classification and probabilities (with scaling applied).")