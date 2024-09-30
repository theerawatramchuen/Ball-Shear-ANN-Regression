import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import and preprocess data (unchanged)
df_org = pd.read_csv('ballshear.csv')
df = pd.read_csv('ballshear.csv')

# Data cleaning and preprocessing (unchanged)
to_drop = ['LSL', 'USL', 'Parameter.Recipe', 'PROJECT_TYPE']
df_org.drop(to_drop, inplace=True, axis=1)
df_org.dropna(inplace=True)
df.drop(to_drop, inplace=True, axis=1)
df.dropna(inplace=True)
df_org.dropna(inplace=False)

# Drop unwanted columns (unchanged)
to_drop = ['C_RISTIC', 'DATE_TIME', 'SHIFT', 'PT', 'EN_NO', 'DEVICE', 'REMARK', 'SD',
           'BOM_NO', 'SUBGRP', 'PLANT_ID', 'MC_ID', 'MC_NO', 'COUNTER',
           'CHAR_MINOR', 'PACKAGE', 'DATE_', 'CIMprofile.cim_machine_name',
           'Parameter.DataType', 'Parameter.Unit',
           'Parameter.Valid', 'Parameter.EquipOpn', 'Parameter.EquipID',
           'Parameter.ULotID', 'Parameter.CreateTime']
df.drop(to_drop, inplace=True, axis=1)

# Rearrange column position (unchanged)
cols_to_move = ['Parameter.Max', 'Parameter.Min', 'Parameter.Value', 'MEANX']
new_cols = np.hstack((df.columns.difference(cols_to_move), cols_to_move))
df = df.reindex(columns=new_cols)

# One-hot encoding (unchanged)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [0, 1, 2, 3, 4, 5])], remainder='passthrough')
X = ct.fit_transform(df).toarray()

# Split X to 2 arrays as x inputFeatures and y outputResponse
selector = [i for i in range(X.shape[1]) if i != 47]  # Column 47 is 'MEANX'
x = X[:, selector]
y = X[:, 47]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Convert to PyTorch tensors and move to GPU
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# Define the PyTorch model
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 47)
        self.fc2 = nn.Linear(47, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model and move it to GPU
input_size = X_train.shape[1]
model = ANN(input_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Move batch to GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'ballshear_model.pth')

# Predicting the results of the Test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Move predictions back to CPU for numpy operations
y_pred_np = y_pred.cpu().numpy()
y_test_np = y_test.cpu().numpy()

# Compute MAPE
MAPE = np.mean(100 * (np.abs(y_test_np - y_pred_np) / y_test_np))
print(f'Accuracy: {100 - MAPE:.2f}%')

# Create comparison DataFrame
compare = pd.DataFrame({'y_test': y_test_np.flatten(), 'y_pred': y_pred_np.flatten()})
compare.to_csv('compare.csv', index=False)

# Predict on the entire dataset
x_tensor = torch.FloatTensor(x).to(device)
model.eval()
with torch.no_grad():
    PRED = model(x_tensor)

# Move predictions back to CPU and add to the original DataFrame
df['PRED'] = PRED.cpu().numpy().flatten()

# Export output to csv
df.to_csv('ballshear_regression.csv', index=False)