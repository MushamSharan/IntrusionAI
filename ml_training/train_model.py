import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('logs/packets_features.csv')

# Preprocessing
label_encoder = LabelEncoder()
data['protocol'] = label_encoder.fit_transform(data['protocol'])
data['label'] = data['label'].apply(lambda x: 1 if x == 'Attack' else 0)

# Features and labels
features = data[['packet_length', 'src_port', 'dst_port', 'flags', 'protocol']]
labels = data['label']

# Reshape the features to match CNN input format (batch, channels, height, width)
features = features.values.reshape(features.shape[0], 1, features.shape[1], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# CNN Model
class NIDS_CNN(nn.Module):
    def __init__(self):
        super(NIDS_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))  # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))  # Conv layer 2
        self.fc1 = nn.Linear(64 * 1 * 1, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU activation
        x = torch.max_pool2d(x, 2)  # Pooling layer
        x = torch.relu(self.conv2(x))  # Apply ReLU activation
        x = torch.max_pool2d(x, 2)  # Pooling layer
        x = x.view(-1, 64 * 1 * 1)  # Flatten the output
        x = torch.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Instantiate model, loss function and optimizer
model = NIDS_CNN()
criterion = nn.CrossEntropyLoss()  # Since it's a classification problem
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
