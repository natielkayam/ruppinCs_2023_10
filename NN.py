import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



df = pd.read_csv("Alerts_for_model.csv", low_memory=False)
df.drop(columns="Unnamed: 0", inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
hour_categories = pd.cut(df['hour'], bins=[00, 6, 10, 15, 20, 24], labels=['Night', 'MorningRush', 'Noon', 'EveningRush', 'Night'], ordered=False)
# Add the hour categories as a new column in the DataFrame
df['hour_category'] = hour_categories
df.loc[df['hour_category'].isna(), 'hour_category'] = 'Night'

Night = df[df['hour_category'] == 'Night']
Night.reset_index(drop=True, inplace=True)
MorningRush = df[df['hour_category'] == 'MorningRush']
MorningRush.reset_index(drop=True, inplace=True)
Noon = df[df['hour_category'] == 'Noon']
Noon.reset_index(drop=True, inplace=True)
EveningRush = df[df['hour_category'] == 'EveningRush']
EveningRush.reset_index(drop=True, inplace=True)


def NN(df):
    X = df[['longitude', 'latitude', 'Jam_Level', 'Rating', 'Max_Reliability', 'street_frequency_bin', 'month', 'day', 'hour','day_name', 'cluster_label', 'season', 'holiday']]
    # Create input and output DataFrames
    y = pd.DataFrame(df['Target'])
    y['Target'] = y['Target'].replace({True: 1, False: 0})
    # Convert object type columns to appropriate types
    X.loc[:, 'street_frequency_bin'] = X['street_frequency_bin'].replace({'Very Low' : 1, 'Low' : 2, 'Medium' : 3, 'High' : 4, 'Very High' : 5})
    X.loc[:, 'cluster_label'] = X['cluster_label'].astype('int64')
    X.loc[:, 'Rating'] = X['Rating'].astype('int64')
    X.loc[:, 'hour'] = X['hour'].astype('int64')
    X.loc[:, 'day'] = X['day'].astype('int64')
    X.loc[:, 'month'] = X['month'].astype('int64')
    X.loc[:, 'month'] = X['month'].astype('int64')
    X.loc[:, 'Max_Reliability'] = X['Max_Reliability'].astype('int64')
    X.loc[:, 'longitude'] = X['longitude'].astype('float64')
    X.loc[:, 'latitude'] = X['latitude'].astype('float64')
    X['Jam_Level'] = X['Jam_Level'].replace({'NO_SUBTYPE': 1, 'JAM_MODERATE_TRAFFIC': 2, 'JAM_HEAVY_TRAFFIC': 3, 'JAM_STAND_STILL_TRAFFIC': 4})
    X.loc[:, 'day_name'] = X['day_name'].replace({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})
    X.loc[:, 'holiday'] = X['holiday'].replace({True: 1, False: 0})
    X.loc[:, 'season'] = X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})


    class BinaryClassifier(nn.Module):
        def __init__(self, input_size):
            super(BinaryClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.3)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation.values, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation.values, dtype=torch.float32)

    input_size = X_train.shape[1]
    model = BinaryClassifier(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    epochs = 10
    batch_size = 8

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_train_tensor[i:i + batch_size]
            targets = y_train_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model on train set
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predictions = (train_outputs >= 0.5).squeeze().numpy().astype(int)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_cm = confusion_matrix(y_train, train_predictions)

    # Evaluate the model on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).squeeze().numpy().astype(int)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_cm = confusion_matrix(y_test, test_predictions)

    print("Train Accuracy:", train_accuracy)
    print("Train Confusion Matrix:")
    print(train_cm)
    print("Test Accuracy:", test_accuracy)
    print("Test Confusion Matrix:")
    print(test_cm)

print("Morning Rush: ")
NN(MorningRush)
print("Noon: ")
NN(Noon)
print("EveningRush: ")
NN(EveningRush)
print("Night: ")
NN(Night)