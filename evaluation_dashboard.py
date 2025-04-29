import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import plotly.express as px

st.title("AI-NIDS Model Evaluation Dashboard")

# Load your trained model
try:
    model = joblib.load('ml_training/trained_xgboost_model.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load your test data and true labels for evaluation
try:
    X_test = pd.read_csv('data/kaggle_test_features.csv')
    y_test = pd.read_csv('data/kaggle_test_labels.csv')
    y_test = y_test.iloc[:, 0]  # Select the first column to make it a Series
except Exception as e:
    st.error(f"Error loading test data: {e}")
    X_test = None
    y_test = None

if model and X_test is not None and y_test is not None:
    # Make predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    class_names = [str(i) for i in np.unique(y_test)]
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test)) # Ensure labels match
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=class_names,
                       y=class_names,
                       color_continuous_scale='Blues')
    st.plotly_chart(fig_cm)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Handle potential division by zero
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

else:
    st.warning("Model or test data not loaded.")