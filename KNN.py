import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance


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

print(df['street_frequency_bin'].unique())

# Fit the KNN model
def KNN(df):
    # X = df[['longitude', 'latitude', 'Jam_Level', 'Rating', 'Max_Reliability', 'street_frequency_bin', 'month', 'day', 'hour','day_name', 'cluster_label', 'season', 'holiday']]
    X = df[['street_frequency_bin', 'month', 'day', 'hour', 'cluster_label','hour_category']]
    # 'day_name'
    # Create input and output DataFrames
    y = pd.DataFrame(df['Target'])
    y['Target'] = y['Target'].replace({True: 1, False: 0})
    # Convert object type columns to appropriate types
    X.loc[:, 'street_frequency_bin'] = X['street_frequency_bin'].replace({'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5})
    X.loc[:, 'hour_category'] = X['hour_category'].replace({'MorningRush': 1, 'Noon': 2, 'EveningRush': 3, 'Night': 4})
    X.loc[:, 'cluster_label'] = X['cluster_label'].astype('int64')
    # X.loc[:, 'Rating'] = X['Rating'].astype('int64')
    X.loc[:, 'hour'] = X['hour'].astype('int64')
    X.loc[:, 'day'] = X['day'].astype('int64')
    X.loc[:, 'month'] = X['month'].astype('int64')
    # X.loc[:, 'Max_Reliability'] = X['Max_Reliability'].astype('int64')
    # X.loc[:, 'longitude'] = X['longitude'].astype('float64')
    # X.loc[:, 'latitude'] = X['latitude'].astype('float64')
    # X['Jam_Level'] = X['Jam_Level'].replace({'NO_SUBTYPE': 1, 'JAM_MODERATE_TRAFFIC': 2, 'JAM_HEAVY_TRAFFIC': 3, 'JAM_STAND_STILL_TRAFFIC': 4})
    # X.loc[:, 'day_name'] = X['day_name'].replace({'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7})
    # X.loc[:, 'holiday'] = X['holiday'].replace({True: 1, False: 0})
    # X.loc[:, 'season'] = X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)

    # Perform undersampling on the training set
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Build and fit KNN model
    k_values = range(1, 21)
    accuracy_values = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_resampled, y_train_resampled)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_values.append(accuracy)

    # Find the optimal value of K
    optimal_k = np.argmax(accuracy_values) + 1

    # Build the final model with the optimal K
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)

    # Generate predictions on the test set
    y_pred_prob = knn.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC: {auc:.3f}")

    # Evaluate the model
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    # Generate predictions on the test set
    y_pred = knn.predict(X_test)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)
    print(f"Train score: {train_score:.3f}, Test score: {test_score:.3f}")



print("all df :")
KNN(df)
print("Morning Rush: ")
KNN(MorningRush)
print("Noon: ")
KNN(Noon)
print("EveningRush: ")
KNN(EveningRush)
print("Night: ")
KNN(Night)
