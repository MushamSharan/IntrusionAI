# preprocess_data.py
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json

# Define the path to the directory containing the CSV files
data_directory = '/Users/mushamsharan/Desktop/Ai-NIDS/nids_project/data/DataSet'
all_files = glob.glob(os.path.join(data_directory, "*.csv"))
all_df = []

for f in all_files:
    try:
        df = pd.read_csv(f)
        print(f"Loaded: {f}, shape: {df.shape}")
        all_df.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

if all_df:
    combined_df = pd.concat(all_df, ignore_index=True)
    print(f"\nCombined DataFrame shape (initial): {combined_df.shape}")

    # Identify and drop rows with infinite values
    inf_rows_before = len(combined_df)
    combined_df = combined_df[~combined_df.isin([float('inf'), float('-inf')]).any(axis=1)]
    inf_rows_dropped = inf_rows_before - len(combined_df)
    print(f"Number of rows dropped due to infinite values: {inf_rows_dropped}")
    print(f"Combined DataFrame shape after dropping infinite value rows: {combined_df.shape}")

    # Drop rows with any remaining NaN values
    nan_rows_before = len(combined_df)
    combined_df.dropna(inplace=True)
    nan_rows_dropped = nan_rows_before - len(combined_df)
    print(f"Number of rows dropped due to NaN values: {nan_rows_dropped}")
    print(f"Combined DataFrame shape after dropping NaN rows: {combined_df.shape}")

    if combined_df.empty:
        print("\nERROR: DataFrame is empty after handling. Further investigation of data is needed.")
    else:
        label_encoder = LabelEncoder()
        combined_df['label_multiclass'] = label_encoder.fit_transform(combined_df[' Label'])
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("\nLabel Mapping for Multi-Class:")
        print(label_mapping)
        print("\nValue counts for multi-class labels:")
        print(combined_df['label_multiclass'].value_counts())

        X = combined_df.drop([' Label', 'label_multiclass'], axis=1, errors='ignore')
        y = combined_df['label_multiclass']
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        X.columns = X.columns.str.strip()
        numerical_features = [col.strip() for col in numerical_features]

        # Save the list of numerical features
        with open('../data/numerical_features.json', 'w') as f:
            json.dump(numerical_features, f)
        print("List of numerical features saved to ../data/numerical_features.json")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[numerical_features])
        X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)
        X_processed = pd.concat([X_scaled_df, X.drop(columns=numerical_features, errors='ignore').reset_index(drop=True)], axis=1)
        y_processed = combined_df['label_multiclass']

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed)

        # Ensure the '../data' directory exists
        os.makedirs('../data', exist_ok=True)
        X_train.to_csv('../data/kaggle_train_features.csv', index=False)
        y_train.to_csv('../data/kaggle_train_labels.csv', index=False)
        X_test.to_csv('../data/kaggle_test_features.csv', index=False)
        y_test.to_csv('../data/kaggle_test_labels.csv', index=False)

        print("\nProcessed data saved to ../data/kaggle_train_features.csv, ../data/kaggle_train_labels.csv, ../data/kaggle_test_features.csv, and ../data/kaggle_test_labels.csv")

        # Save the scaler in the '../data' directory
        joblib.dump(scaler, '../data/scaler.joblib')
        print("Scaler saved to ../data/scaler.joblib")

else:
        print("Error: No CSV files were loaded.")