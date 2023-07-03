import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

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

def xgboost(df):
    #X = df[['longitude', 'latitude', 'Jam_Level', 'Rating', 'Max_Reliability', 'street_frequency_bin', 'month', 'day', 'hour','day_name', 'cluster_label', 'season', 'holiday']]
    X = df[[ 'street_frequency_bin', 'month', 'day', 'hour', 'cluster_label','hour_category']]
    #'day_name'
    # Create input and output DataFrames
    y = pd.DataFrame(df['Target'])
    y['Target'] = y['Target'].replace({True: 1, False: 0})
    # Convert object type columns to appropriate types
    X.loc[:, 'street_frequency_bin'] = X['street_frequency_bin'].replace({'Very Low' : 1, 'Low' : 2, 'Medium' : 3, 'High' : 4, 'Very High' : 5})
    X.loc[:, 'cluster_label'] = X['cluster_label'].astype('int64')
    X.loc[:, 'hour_category'] = X['hour_category'].replace({'MorningRush': 1, 'Noon': 2, 'EveningRush': 3, 'Night': 4})
    #X.loc[:, 'Rating'] = X['Rating'].astype('int64')
    X.loc[:, 'hour'] = X['hour'].astype('int64')
    X.loc[:, 'day'] = X['day'].astype('int64')
    X.loc[:, 'month'] = X['month'].astype('int64')
    #X.loc[:, 'Max_Reliability'] = X['Max_Reliability'].astype('int64')
    #X.loc[:, 'longitude'] = X['longitude'].astype('float64')
    #X.loc[:, 'latitude'] = X['latitude'].astype('float64')
    #X['Jam_Level'] = X['Jam_Level'].replace({'NO_SUBTYPE': 1, 'JAM_MODERATE_TRAFFIC': 2, 'JAM_HEAVY_TRAFFIC': 3, 'JAM_STAND_STILL_TRAFFIC': 4})
    #X.loc[:, 'day_name'] = X['day_name'].replace({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})
    #X.loc[:, 'holiday'] = X['holiday'].replace({True: 1, False: 0})
    #X.loc[:, 'season'] = X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
    # Normalize the input data using MinMaxScaler
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    rus = RandomUnderSampler(random_state=22)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    # Perform undersampling on the test set
    X_test_resampled, y_test_resampled = rus.fit_resample(X_test, y_test)

    xgb_model = xgb.XGBClassifier()

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    train_score = xgb_model.score(X_train_resampled, y_train_resampled)
    test_score = xgb_model.score(X_test, y_test)

    y_train_pred = xgb_model.predict(X_train_resampled)
    y_test_pred = xgb_model.predict(X_test_resampled)


    # Calculate the confusion matrices
    train_cm = confusion_matrix(y_train_resampled, y_train_pred)
    test_cm = confusion_matrix(y_test_resampled, y_test_pred)

    print(f"train score : {train_score} test score : {test_score}")
    print(f"Train score: {train_score}, Test score: {test_score}")
    print("Train Confusion Matrix:")
    print(train_cm)
    print("Test Confusion Matrix:")
    print(test_cm)

print("morning rush : ")
xgboost(MorningRush)
print("Noon : ")
xgboost(Noon)
print("Evning rush : ")
xgboost(EveningRush)
print("Night : ")
xgboost(Night)