# In train_xgboost_model_v2.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib # Make sure joblib is imported if you haven't already

# Load your training and testing data
X_train = pd.read_csv('../data/kaggle_train_features.csv')
y_train = pd.read_csv('../data/kaggle_train_labels.csv').squeeze() # Use squeeze() to get a Series
X_test = pd.read_csv('../data/kaggle_test_features.csv')
y_test = pd.read_csv('../data/kaggle_test_labels.csv').squeeze() # Use squeeze() to get a Series

# Initialize and train XGBoost (as before)
num_classes = len(pd.unique(y_train))
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    random_state=42
)
xgb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model (as before)
accuracy = accuracy_score(y_test, y_pred)
print(f"Multi-Class Accuracy: {accuracy:.4f}")
print("\nMulti-Class Classification Report:")
print(classification_report(y_test, y_pred))

# Generate and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = [str(i) for i in range(num_classes)] # Get class names (you might want to map back to original labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model (as before)
joblib.dump(xgb_classifier, 'trained_xgboost_model.joblib')
print("Trained model saved as trained_xgboost_model.joblib")