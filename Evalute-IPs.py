import pandas as pd

# Load your dataset
data = pd.read_csv('logs/packets_features.csv')

# View the first few rows to check the data (optional)
print(data.head())

# Manually add the label column based on src_ip
# Here, we're labeling traffic from '192.168.1.100' as attack (1), others as normal (0)
data['label'] = data['src_ip'].apply(lambda x: 1 if x == '192.168.1.100' else 0)

# Verify that the label column is added
print(data.head())
