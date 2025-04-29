# IntrusionAI
IntrusionAI: AI-Driven Network Intrusion Detection System
Overview
IntrusionAI is an AI-driven Network Intrusion Detection System (NIDS) that leverages machine learning to identify potential security threats. By employing an XGBoost model, IntrusionAI analyzes network traffic data to detect anomalies and potential intrusions in real-time. This project aims to provide an intelligent and automated approach to network security monitoring.

Features
AI-Powered Detection: Utilizes an XGBoost model for accurate classification of network intrusions, including DDoS, PortScan, and Botnet attacks.

Interactive Dashboard: A Streamlit dashboard provides a user-friendly interface for visualizing model predictions and probabilities.

Simplified Input: Users can input network traffic features to receive immediate threat assessments.

Dockerized Deployment: The application is packaged using Docker for easy deployment and portability.

Technologies Used
Python
Pandas
Scikit-learn
XGBoost
Joblib
Streamlit
NumPy
Plotly
Docker
JSON

Setup Instructions
Prerequisites:
Python 3.9 or higher
Docker

Clone the repository:
git clone <repository_url>
cd Ai-NIDS

Install dependencies:
pip install -r requirements.txt

Data Preparation:
Download the network traffic data and place it in the data/DataSet directory.
Run the preprocess_data.py script to preprocess the data:
python ml_training/preprocess_data.py

Build the Docker image:
docker build -t ai-nids-simple-dashboard .
Run the Docker container:
docker run -p 8501:8501 ai-nids-simple-dashboard

Access the dashboard:
Open your web browser and go to http://localhost:8501.

Current Status and Future Development
IntrusionAI is currently under development. The following key tasks are in progress:
Resolving Docker build errors.
Enhancing the dashboard with more meaningful feature inputs.
Exploring model evaluation metrics.
Investigating real-time data integration.

Future development plans include:
Implementing feature importance analysis.
Adding more sophisticated visualizations.
Integrating with network monitoring tools.
Developing alerting mechanisms.
Exploring other machine learning models.

Contributions
Contributions are welcome! If you have any ideas or suggestions for improving IntrusionAI, please feel free to submit a pull request or open an issue.

Created by
Musham Sharan.

Document Link : https://docs.google.com/document/d/1kdrePZMsBuZ8J4sC__2V3y_E-o589jyIw-Vgil0iLE4/edit?tab=t.0
