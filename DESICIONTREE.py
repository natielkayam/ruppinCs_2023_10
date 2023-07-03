import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import compute_class_weight

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


def dt(df):
    X = df[['Count', 'longitude', 'latitude', 'Jam_Level', 'Rating', 'Max_Reliability', 'street_frequency_bin', 'month', 'day', 'hour','day_name', 'cluster_label', 'season', 'holiday']]
    #X = df[[ 'count','street_frequency_bin', 'month', 'day', 'hour', 'cluster_label','hour_category']]
    #'day_name'
    # Create input and output DataFrames
    y = pd.DataFrame(df['Target'])
    y['Target'] = y['Target'].replace({True: 1, False: 0})
    # Convert object type columns to appropriate types
    X.loc[:, 'street_frequency_bin'] = X['street_frequency_bin'].replace({'Very Low' : 1, 'Low' : 2, 'Medium' : 3, 'High' : 4, 'Very High' : 5})
    X.loc[:, 'cluster_label'] = X['cluster_label'].astype('int64')
    #X.loc[:, 'hour_category'] = X['hour_category'].replace({'MorningRush': 1, 'Noon': 2, 'EveningRush': 3, 'Night': 4})
    X.loc[:, 'Rating'] = X['Rating'].astype('int64')
    X.loc[:, 'hour'] = X['hour'].astype('int64')
    X.loc[:, 'day'] = X['day'].astype('int64')
    X.loc[:, 'month'] = X['month'].astype('int64')
    X.loc[:, 'Max_Reliability'] = X['Max_Reliability'].astype('int64')
    X.loc[:, 'longitude'] = X['longitude'].astype('float64')
    X.loc[:, 'latitude'] = X['latitude'].astype('float64')
    X['Jam_Level'] = X['Jam_Level'].replace({'NO_SUBTYPE': 1, 'JAM_MODERATE_TRAFFIC': 2, 'JAM_HEAVY_TRAFFIC': 3, 'JAM_STAND_STILL_TRAFFIC': 4})
    X.loc[:, 'day_name'] = X['day_name'].replace({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})
    X.loc[:, 'holiday'] = X['holiday'].replace({True: 1, False: 0})
    X.loc[:, 'season'] = X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
    X.loc[:, 'Count'] = X['Count'].astype('int64')
    #give more significant to
    # Scale the significant features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X[['hour', 'day', 'cluster_label', 'month', 'street_frequency_bin', 'day_name']])
    #
    # X.loc[:, ['hour', 'day', 'cluster_label', 'month', 'street_frequency_bin', 'day_name']] = X_scaled
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rus = RandomUnderSampler(random_state=22)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    # Perform undersampling on the test set
    X_test_resampled, y_test_resampled = rus.fit_resample(X_test, y_test)
    #print(y_train_resampled['Target'].value_counts())
    #print(y_test_resampled['Target'].value_counts())


    # Create a Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()

    # Train the classifier on the resampled data
    dt_classifier.fit(X_train_resampled, y_train_resampled)

    # Find the pruned classifier with the best score #done already
    best_alpha = 5.07872016e-05
    best_classifier = DecisionTreeClassifier(ccp_alpha=best_alpha)
    best_classifier.fit(X_train_resampled, y_train_resampled)

    # Plot the decision tree
    #plt.figure(figsize=(15, 10))
    #tree.plot_tree(best_classifier, feature_names=X.columns, class_names=["False", "True"], filled=True)
    #plt.show()
    # Make predictions on the resampled train and test sets
    y_train_pred = best_classifier.predict(X_train_resampled)
    y_test_pred = best_classifier.predict(X_test_resampled)

    # Calculate the train and test scores
    train_score = accuracy_score(y_train_resampled, y_train_pred)
    test_score = accuracy_score(y_test_resampled, y_test_pred)

    # Calculate the confusion matrices
    train_cm = confusion_matrix(y_train_resampled, y_train_pred)
    test_cm = confusion_matrix(y_test_resampled, y_test_pred)

    print(f"Train score: {train_score}, Test score: {test_score}")
    print("Train Confusion Matrix:")
    print(train_cm)
    print("Test Confusion Matrix:")
    print(test_cm)



print("Morning Rush: ")
dt(MorningRush)
print("Noon: ")
dt(Noon)
print("EveningRush: ")
dt(EveningRush)
print("Night: ")
dt(Night)