import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title("AI-NIDS Simple Detection")

# Load the trained XGBoost model
try:
    model = joblib.load('ml_training/trained_xgboost_model.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load the feature names
try:
    feature_names = pd.read_csv('data/kaggle_train_features.csv', nrows=1).columns.tolist()
    non_numerical_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
    numerical_features = [col for col in feature_names if col not in non_numerical_cols]
    num_numerical_features = len(numerical_features)
except Exception as e:
    st.error(f"Error loading feature names: {e}")
    feature_names = None
    numerical_features = None
    num_numerical_features = 0

# Load the scaler
try:
    scaler = joblib.load('data/scaler.joblib')
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    scaler = None

# Load the label mapping
label_mapping = {0: 'BENIGN', 1: 'Bot', 2: 'PortScan', 3: 'DDoS', 4: 'Web Attack', 5: 'FTP-Patator', 6: 'SSH-Patator', 7: 'DoS slowloris', 8: 'DoS Slowhttptest', 9: 'DoS Hulk', 10: 'DoS GoldenEye', 11: 'Heartbleed', 12: 'Infiltration', 13: 'Invalid Source', 14: 'Exploits'}

st.sidebar.header("Enter Flow Duration")
flow_duration_input = st.sidebar.number_input("Flow Duration", value=0.0)

if st.sidebar.button("Predict"):
    if model and feature_names and scaler and numerical_features:
        try:
            # Create a dummy input array with zeros for all numerical features
            dummy_numerical_input = np.zeros(len(numerical_features))

            # Find the index of 'Flow Duration' within the numerical features
            if 'Flow Duration' in numerical_features:
                flow_duration_index_in_numerical = numerical_features.index('Flow Duration')
                dummy_numerical_input[flow_duration_index_in_numerical] = flow_duration_input

            # Reshape for scaling
            dummy_numerical_input_reshaped = dummy_numerical_input.reshape(1, -1)

            # Scale the dummy numerical input
            scaled_dummy_numerical_input = scaler.transform(dummy_numerical_input_reshaped)

            # Create a full dummy input array with zeros for all features
            full_dummy_input = np.zeros(len(feature_names))

            # Place the scaled 'Flow Duration' value in the correct position in the full input
            if 'Flow Duration' in feature_names:
                flow_duration_original_index = feature_names.index('Flow Duration')
                # Find the index of 'Flow Duration' in the numerical features list
                if 'Flow Duration' in numerical_features:
                    flow_duration_scaled_index = numerical_features.index('Flow Duration')
                    # Find the corresponding index in the full feature list
                    numerical_indices_in_full = [feature_names.index(f) for f in numerical_features]
                    full_dummy_input[numerical_indices_in_full[flow_duration_scaled_index]] = scaled_dummy_numerical_input[0][flow_duration_scaled_index]


            prediction_probabilities = model.predict_proba(full_dummy_input.reshape(1, -1))[0]
            predicted_numerical = np.argmax(prediction_probabilities)
            predicted_label = label_mapping.get(predicted_numerical, 'UNKNOWN')

            st.subheader("Prediction:")
            st.markdown(f"<p style='font-size: 24px; font-weight: bold;'>Predicted Traffic: <span style='color: green;'>{predicted_label}</span></p>", unsafe_allow_html=True)
            st.write(f"Class ID: {predicted_numerical}")

            prob_df = pd.DataFrame({'Class': list(label_mapping.values()), 'Probability': prediction_probabilities})
            fig_prob = px.bar(prob_df, x='Class', y='Probability', title='Prediction Probabilities')
            st.plotly_chart(fig_prob)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Model, feature names, or scaler not loaded.")

st.markdown("---")
st.info("Enter a value for 'Flow Duration' and click 'Predict'. Note: This is a simplified example.")