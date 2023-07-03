import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

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

# num1 = df[df['Target'] == 1]
# num0 = df[df['Target'] == 0]
# print(f"{num1} alerts with 1")
# print(f"{num0} alerts with 0")

#
# def train_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs, patience=10):
#     train_losses = []
#     val_losses = []
#     best_val_loss = np.inf
#     epochs_no_improve = 0
#
#     for epoch in range(epochs):
#         # Train the model for one epoch
#         history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val), verbose=0)
#
#         # Calculate training and validation loss
#         train_loss = history.history['loss'][0]
#         val_loss = history.history['val_loss'][0]
#
#         # Save the training and validation losses
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#
#         # Check for improvement in validation loss
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered - training stopped.")
#                 break
#
#     # Plot the learning curve
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
#
#     return train_losses, val_losses

def hst_lstm(df):
    # Extract the relevant features
    X_spatial = df[['longitude', 'latitude']].copy()
    X_temporal = df[['Count', 'Jam_Level', 'Rating', 'Max_Reliability', 'street_frequency_bin', 'month', 'day', 'hour', 'day_name','cluster_label', 'season', 'holiday','hour_category']].copy()
    # Convert object type columns to appropriate types
    X_temporal.loc[:, 'street_frequency_bin'] = X_temporal['street_frequency_bin'].replace({'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5})
    X_temporal.loc[:, 'cluster_label'] = X_temporal['cluster_label'].astype('int64')
    X_temporal.loc[:, 'hour_category'] = X_temporal['hour_category'].replace({'MorningRush': 1, 'Noon': 2, 'EveningRush': 3, 'Night': 4})
    X_temporal.loc[:, 'Rating'] = X_temporal['Rating'].astype('int64')
    X_temporal.loc[:, 'hour'] = X_temporal['hour'].astype('int64')
    X_temporal.loc[:, 'day'] = X_temporal['day'].astype('int64')
    X_temporal.loc[:, 'month'] = X_temporal['month'].astype('int64')
    X_temporal.loc[:, 'Max_Reliability'] = X_temporal['Max_Reliability'].astype('int64')
    # X_temporal.loc[:, 'longitude'] = X_temporal['longitude'].astype('float64')
    # X_temporal.loc[:, 'latitude'] = X_temporal['latitude'].astype('float64')
    X_temporal['Jam_Level'] = X_temporal['Jam_Level'].replace({'NO_SUBTYPE': 1, 'JAM_MODERATE_TRAFFIC': 2, 'JAM_HEAVY_TRAFFIC': 3, 'JAM_STAND_STILL_TRAFFIC': 4})
    X_temporal.loc[:, 'day_name'] = X_temporal['day_name'].replace({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})
    X_temporal.loc[:, 'holiday'] = X_temporal['holiday'].replace({True: 1, False: 0})
    X_temporal.loc[:, 'season'] = X_temporal['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
    X_temporal.loc[:, 'Count'] = X_temporal['Count'].astype('int64')
    y = df['Target'].astype(int)

    # Scale the features
    scaler = StandardScaler()
    X_spatial_scaled = scaler.fit_transform(X_spatial)
    X_temporal_scaled = scaler.fit_transform(X_temporal)

    # # Split the data into training and testing sets
    # X_spatial_train, X_spatial_test, X_temporal_train, X_temporal_test, y_train, y_test = train_test_split(
    #     X_spatial_scaled, X_temporal_scaled, y, test_size=0.2, random_state=22)
    # Create an instance of StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=22)

    # Split the data
    for train_index, test_index in splitter.split(X_spatial_scaled, y):
        # Split spatial features
        X_spatial_train = X_spatial_scaled[train_index]
        X_spatial_test = X_spatial_scaled[test_index]

        # Split temporal features
        X_temporal_train = X_temporal_scaled[train_index]
        X_temporal_test = X_temporal_scaled[test_index]

        # Split target variable
        y_train = y[train_index]
        y_test = y[test_index]

    # Reshape the temporal input to include an additional dimension for timesteps
    X_temporal_train = np.expand_dims(X_temporal_train, axis=1)
    X_temporal_test = np.expand_dims(X_temporal_test, axis=1)

    # Define the LSTM architecture
    spatial_input = Input(shape=(X_spatial_train.shape[1],))
    temporal_input = Input(shape=(X_temporal_train.shape[1], X_temporal_train.shape[2]))

    spatial_branch = Dense(32, activation='sigmoid')(spatial_input)
    temporal_branch = LSTM(32)(temporal_input)

    merged = Concatenate()([spatial_branch, temporal_branch])
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[spatial_input, temporal_input], outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #train_with_early_stopping(model, [X_spatial_train, X_temporal_train], y_train, [X_spatial_test, X_temporal_test], y_test, epochs=2000, patience=10)
    # Train the model
    model.fit([X_spatial_train, X_temporal_train], y_train, batch_size=4, epochs=50, verbose=1)

    # Evaluate the model
    _, train_accuracy = model.evaluate([X_spatial_train, X_temporal_train], y_train, verbose=0)
    _, test_accuracy = model.evaluate([X_spatial_test, X_temporal_test], y_test, verbose=0)
    # Make predictions
    y_train_pred = model.predict([X_spatial_train, X_temporal_train])
    y_test_pred = model.predict([X_spatial_test, X_temporal_test])

    # Convert predictions to binary classes (0 or 1)
    y_train_pred_classes = np.round(y_train_pred).flatten()
    y_test_pred_classes = np.round(y_test_pred).flatten()

    # Create confusion matrices
    train_confusion_matrix = confusion_matrix(y_train, y_train_pred_classes)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred_classes)

    print("Train Confusion Matrix:")
    print(train_confusion_matrix)
    print()

    print("Test Confusion Matrix:")
    print(test_confusion_matrix)
    print()

    print(f"Train accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")


# Call the HST-LSTM function with your data
hst_lstm(df)
print("morning rush : ")
hst_lstm(MorningRush)
print("Noon : ")
hst_lstm(Noon)
print("Evning rush : ")
hst_lstm(EveningRush)
print("Night : ")
hst_lstm(Night)
