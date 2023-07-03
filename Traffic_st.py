import numpy as np
import pandas as pd
import geopandas as gpd

Traffic_df = pd.read_csv('DataWazeTraffic.csv',low_memory=False)
Traffic_df.columns = ['TS', 'TrafficID', 'PubDate', 'Linqmap_Type', 'GeoRss_Line', 'Linqmap_Speed', 'Linqmap_SpeedKMH',
                      'Linqmap_Length', 'Linqmap_Delay', 'Linqmap_Street', 'Linqmap_City', 'Linqmap_Country',
                      'Linqmap_RoadType', 'Linqmap_StartNode', 'Linqmap_EndNode', 'Linqmap_Level', 'Linqmap_Uuid',
                      'Linqmap_TurnLine', 'Linqmap_TurnType', 'Linqmap_BlockingAlertUuid', 'Waze_geom']

Traffic_df["Linqmap_BlockingAlertUuid"]= Traffic_df["Linqmap_BlockingAlertUuid"].fillna(0)
Traffic_df["Linqmap_BlockingAlertUuid"]=Traffic_df["Linqmap_BlockingAlertUuid"].apply(lambda x: 1 if x!=0 else 0)
Traffic_df.drop(index=Traffic_df[Traffic_df['Linqmap_BlockingAlertUuid'] == 1].index, inplace=True)
Traffic_df.drop(columns='Linqmap_BlockingAlertUuid',inplace = True)
Traffic_df["TS"] = pd.to_datetime(Traffic_df["TS"])
Traffic_df["PubDate"] = pd.to_datetime(Traffic_df["PubDate"])
Traffic_df.query("Linqmap_RoadType == 2", inplace=True)

#TODO: add duration for each Jam(UUID,PUBDATE)
def calculate_duration(group):
    latest_TS = pd.to_datetime(group["TS"].max())
    earliest_TS = pd.to_datetime(group["TS"].min())
    pub_date = pd.to_datetime(group["PubDate"].iloc[0])

    Bias = pd.Timedelta(0)
    if pub_date > earliest_TS:
        Bias = pub_date - earliest_TS

    duration = (latest_TS - pub_date) + Bias
    group["duration"] = duration
    return group


groups = Traffic_df.groupby(["Linqmap_Uuid", "PubDate"], group_keys=True)
Traffic_df = groups.apply(calculate_duration)


Traffic_df.drop(
    columns=["TS", "Linqmap_Type", "Linqmap_Speed", "Linqmap_City", "Linqmap_Country",
             "Linqmap_RoadType", "Linqmap_StartNode", "Linqmap_EndNode", "Linqmap_TurnLine", "Linqmap_TurnType",
             "Waze_geom", "TrafficID"], inplace=True)
Traffic_df.reset_index(drop=True,inplace=True)

#TODO:groupby for each unique jam(space and time)
groups_v2 = Traffic_df.groupby(["Linqmap_Uuid","PubDate","GeoRss_Line"])

merged = groups_v2.apply(lambda x: x.fillna(method='ffill').iloc[-1])
# Add the count column
merged["Count"] = groups_v2.size().values
# Add the max and min_max_reliability columns
merged["Max_Linqmap_Level"] = groups_v2["Linqmap_Level"].max().values
merged["Min_Linqmap_Level"] = groups_v2["Linqmap_Level"].min().values
merged["Linqmap_SpeedKMH"] = groups_v2["Linqmap_SpeedKMH"].mean()
merged["Linqmap_Delay"] = groups_v2["Linqmap_Delay"].mean()

Traffic_Filttered = merged
Traffic_Filttered.reset_index(drop=True, inplace=True)



a = np.array(Traffic_Filttered["GeoRss_Line"])
latlan = []
for i in a:
    word = i.split()
    w = ""
    tmp = []
    for i in range(0, len(word), 2):
        w = [float(word[i]), float(word[i + 1])]
        tmp.append(w)
    latlan.append(tmp)
Traffic_Filttered['latlan'] = latlan
from shapely import LineString

geo = []
for i in Traffic_Filttered['latlan']:
    a = LineString(i)
    geo.append(a)
Traffic_Filttered['TrafficLinestring'] = geo
gdfTraffic_df = gpd.GeoDataFrame(Traffic_Filttered, geometry=Traffic_Filttered["TrafficLinestring"], crs="EPSG:4326")
Traffic_Filttered = gdfTraffic_df

Traffic_Filttered.drop(columns=['latlan',"GeoRss_Line"],inplace=True)

street_frequencies = Traffic_Filttered['Linqmap_Street'].value_counts()
print(street_frequencies)

# Extract the street names and frequencies from the provided data
streets = Traffic_Filttered['Linqmap_Street'].unique()
import matplotlib.pyplot as plt

# Define the frequency bins
bins = [0, 300, 750 , 2800, 7000, float('inf')]

# Group the streets into the frequency bins
frequency_bins = pd.cut(street_frequencies, bins=bins)

# Count the number of streets in each frequency bin
bin_counts = frequency_bins.value_counts().sort_index()

# Convert the index to strings
bin_labels = [str(bin) for bin in bin_counts.index]

# # Create the bar plot
# bar1 = plt.bar(bin_labels, bin_counts.values)
# plt.xlabel('Frequency Bins')
# plt.ylabel('Count')
# plt.title('Street Frequencies Histogram')
# plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
# plt.tight_layout()  # Adjust the layout for better spacing
# for rect in bar1:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
#
# plt.show()

Traffic_Filttered['street_frequency'] = Traffic_Filttered['Linqmap_Street'].map(Traffic_Filttered['Linqmap_Street'].value_counts())
# Define the bin edges based on the frequency distribution
bins = [0, 6000, 9000, 12500, float('inf')]
labels = ['Low', 'Medium', 'High', 'Very High']
# Categorize the street frequencies into bins
Traffic_Filttered['street_frequency_bin'] = pd.cut(Traffic_Filttered['street_frequency'], bins=bins, labels=labels)


print(Traffic_Filttered['street_frequency_bin'].value_counts())


Traffic_Filttered["PubDate"] = pd.to_datetime(Traffic_Filttered["PubDate"])
#Traffic_Filttered['duration'] = Traffic_Filttered['duration'] + 0.5*(Traffic_Filttered['duration'])
Traffic_Filttered['threshold_time'] = Traffic_Filttered['PubDate'] - pd.to_timedelta(Traffic_Filttered['duration'])
# jam_start = pd.to_datetime(jam_start)
# alert_start = pd.to_datetime(alert_start)
# duration_jam = pd.to_timedelta(duration_jam)
# threshold_time = jam_start - duration_jam
Traffic_Filttered.to_csv("Traffic_234.csv")
print("save complete")
