import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin_nearest
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

import warnings

Alerts_2304 = pd.read_csv("Alert_2304.csv", low_memory= False)
Traffic_2304 = pd.read_csv("Traffic_234.csv" , low_memory= False)

Alerts_2304.drop(columns="Unnamed: 0", inplace= True)
Traffic_2304.drop(columns="Unnamed: 0", inplace= True)
Alerts_2304.drop(columns="Duration_Alerts", inplace= True)

from shapely.wkt import loads
Traffic_2304['geometry'] = loads(Traffic_2304['geometry'])
Traffic_2304 = gpd.GeoDataFrame(Traffic_2304, geometry='geometry', crs="EPSG:4326")
Alerts_2304['geometry'] = loads(Alerts_2304['geometry'])
Alerts_2304 = gpd.GeoDataFrame(Alerts_2304, geometry='geometry', crs="EPSG:4326")

Alerts_2304["PubDate"] = pd.to_datetime(Alerts_2304["PubDate"])
Alerts_2304['week'] = Alerts_2304["PubDate"].dt.isocalendar().week
Alerts_2304['day'] = Alerts_2304["PubDate"].dt.day
Alerts_2304['hour'] = Alerts_2304["PubDate"].dt.hour

print(Alerts_2304['hour'].unique())

Traffic_2304["PubDate"] = pd.to_datetime(Traffic_2304["PubDate"])
Traffic_2304['week'] = Traffic_2304["PubDate"].dt.isocalendar().week
Traffic_2304['day'] = Traffic_2304["PubDate"].dt.day
Traffic_2304['hour'] = Traffic_2304["PubDate"].dt.hour

print(Traffic_2304['hour'].unique())

def check_alert_during_jam(alert, jam_start, jam_duration):
    alert = pd.to_datetime(alert)
    jam_start = pd.to_datetime(jam_start)
    jam_end = jam_start + pd.to_timedelta(jam_duration)
    return (alert >= jam_start) and (alert <= jam_end)


def check_alert_before_jam(jam_start , duration_jam , alert_start):
    jam_start = pd.to_datetime(jam_start)
    alert_start = pd.to_datetime(alert_start)
    duration_jam = pd.to_timedelta(duration_jam)
    threshold_time = jam_start - duration_jam
    return (threshold_time <= alert_start) and (alert_start < jam_start)


def check_alert(group):
    if any(group['alert_during_jam']) or any(group['alert_before_jam']):
        min_distance_row = group[group['distance'] == group['distance'].min()]
        return min_distance_row
    else:
        random_row = group.sample(n=1)
        return random_row

list_of_weeks = range(1, 53)
list_of_days = range(1, 32)
list_of_hours = range(0, 24)  # Assuming hours range from 0 to 23

list_of_DFweeks = []
for week in list_of_weeks:
    for day in list_of_days:
        for hour in list_of_hours:
            weekTraffic = Traffic_2304[(Traffic_2304['week'] == week) & (Traffic_2304['day'] == day) & (Traffic_2304['hour'] == hour)]
            weekAlerts = Alerts_2304[(Alerts_2304['week'] == week) & (Alerts_2304['day'] == day) & (Alerts_2304['hour'] == hour)]

            # Reset indices of GeoDataFrames
            weekAlerts.reset_index(drop=True, inplace=True)
            weekTraffic.reset_index(drop=True, inplace=True)

            if weekTraffic.empty:
                continue  # Skip iteration if no matching rows in weekTraffic

            # Perform nearest neighbor join
            tmpNERAST = sjoin_nearest(weekAlerts, weekTraffic, how='left')

            # Calculate the distances between geometries
            distances = []
            for _, row in tmpNERAST.iterrows():
                if isinstance(row.geometry, Point):
                    nearest_geom = weekTraffic.loc[row.index_right, 'geometry']
                elif isinstance(row.geometry, LineString):
                    nearest_geom = nearest_points(row.geometry, weekTraffic.loc[row.index_right, 'geometry'])[1]
                else:
                    nearest_geom = None
                if nearest_geom is not None:
                    distance = row.geometry.distance(nearest_geom)
                else:
                    distance = None
                distances.append(distance)
            tmpNERAST['distance'] = distances

            tmpNERAST['alert_during_jam'] = tmpNERAST.apply(
                lambda row: check_alert_during_jam(row['PubDate_left'], row['PubDate_right'], row['duration']), axis=1)

            tmpNERAST['alert_before_jam'] = tmpNERAST.apply(
                lambda row: check_alert_before_jam(row["PubDate_right"], row["duration"], row["PubDate_left"]), axis=1)

            groups = tmpNERAST.groupby(["Linqmap_Uuid_left", "PubDate_left", "Linqmap_Magvar"])

            def check_alert(group):
                if any(group['alert_during_jam']) or any(group['alert_before_jam']):
                    min_distance_row = group[group['distance'] == group['distance'].min()]
                    return min_distance_row
                else:
                    random_row = group.sample(n=1)
                    return random_row

            selected_rows = groups.apply(check_alert).reset_index(drop=True)
            # Filter out any potential duplicate rows
            selected_rows = selected_rows.drop_duplicates(subset=["Linqmap_Uuid_left", "PubDate_left", "Linqmap_Magvar"])
            list_of_DFweeks.append(selected_rows)

Mergedf = pd.concat(list_of_DFweeks)

Mergedf['Target'] = (Mergedf['alert_during_jam'] == 1) | (Mergedf['alert_before_jam'] == 1)

print(len(Mergedf[Mergedf['Target'] == True]))
print(len(Mergedf[Mergedf['Target'] == False]))
import seaborn
correlation = Mergedf.corr()
heatmap = seaborn.heatmap(correlation ,annot = True)
heatmap.set (xlabel = 'x axis',ylabel = 'y axis\t', title = "Correlation matrix of ALERTS\n")
plt.show()
#Save DF to work on model
#columns : PubDate_left , Linqmap_SubType, Linqmap_Street_left, Linqmap_ReportRating, Linqmap_Reliability, Linqmap_Confidence, Count_left, Max_Reliability, Min_Reliability
df = Mergedf[['Count_left', 'PubDate_left', 'Linqmap_SubType', 'Linqmap_Street_left', 'Linqmap_ReportRating', 'street_frequency_bin', 'Linqmap_Reliability', 'Linqmap_Confidence', 'Max_Reliability', 'geometry', 'Min_Reliability', 'Target']].copy()
df.reset_index(drop=True, inplace=True)
df. rename(columns = {'PubDate_left':'Date', 'Linqmap_Street_left':'Street', 'Linqmap_SubType' : 'Jam_Level', 'Linqmap_ReportRating' : 'Rating', 'Linqmap_Reliability' : 'Reliability' ,
                      'Linqmap_Confidence' : 'Confidence', 'Count_left' : 'Count', 'geometry' : 'geo'}, inplace = True)

df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['hour'] = df['Date'].dt.hour
df['day_name'] = pd.to_datetime(df['Date']).dt.strftime('%A')

df.drop(columns=['Confidence', 'Reliability', 'Min_Reliability'], inplace=True)

#ADD HOLIDAY FEATURE
import holidays
IL_holidays = holidays.IL()
def is_holiday(date):
    return date in IL_holidays
df['holiday'] = df['Date'].apply(is_holiday)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
# Add a new column 'season' based on the 'pickupMonth'
df['season'] = df['month'].apply(get_season)
from shapely.geometry import Point

# Define a function to extract longitude and latitude from the geometry
def extract_coordinates(geom):
    point = Point(geom.x, geom.y)
    return point.x, point.y

# Extract longitude and latitude from the geometry column
df['longitude'], df['latitude'] = zip(*df['geo'].apply(lambda geom: extract_coordinates(geom)))

# Perform clustering and assign cluster labels
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=9)  # Choose the appropriate number of clusters
df['cluster_label'] = kmeans.fit_predict(df[['longitude', 'latitude']])

# import seaborn
# correlation = df.corr()
# heatmap = seaborn.heatmap(correlation ,annot = True)
# heatmap.set (xlabel = 'x axis',ylabel = 'y axis\t', title = "Correlation matrix of ALERTS\n")
# plt.show()

print(df['cluster_label'].value_counts())
df.to_csv("Alerts_for_model.csv")

# import folium
# from folium import GeoJson, Popup
# import geopandas as gpd
# import shapely.wkt as wkt
# m = folium.Map(location=[32.794044, 34.989571], zoom_start=12)
# for idx, row in mergebeforeweek2day10withTreshold.iterrows():
#     point = row.geometry
#     popup_text = f"Date: {str(row['PubDate_left'])}<br>Distance: {row['distance']}"
#     popup = folium.Popup(popup_text, max_width=300)
#     folium.Marker(location=[point.x, point.y], popup=popup).add_to(m)
# # Iterate over the GeoDataFrame rows and add linestrings with popups
# for idx, row in mergebeforeweek2day10withTreshold.iterrows():
#     linestring_str = row.TrafficLinestring
#     linestring = wkt.loads(linestring_str)  # Convert the string representation to a LineString object
#     popup_text = f"Date: {str(row['PubDate_right'])}<br>Duration: {row['duration']}<br>Treshold: {row['threshold_time']}"
#     popup = folium.Popup(popup_text, max_width=400)
#     folium.PolyLine(locations=linestring.coords, popup=popup).add_to(m)
#
# # Get the count of rows
# alerts_count = len(mergeduringweek2day10withTreshold.groupby(["Linqmap_Uuid_left", "PubDate_left"], group_keys=True))
# jams_count =  len(mergeduringweek2day10withTreshold.groupby(["Linqmap_Uuid_right", "PubDate_right"], group_keys=True))
#
# # Create the message HTML
# message = f"Alerts: {alerts_count}<br>Jams: {jams_count}"
# message_html = f"""
# <div style='position: fixed;
#             bottom: 50px; left: 50px; width: 150px; height: 80px;
#             background-color: white; border:2px solid grey; z-index:9999;
#             font-size:14px; font-weight:bold;'>
#     {message}
# </div>
# """
#
# # Create a custom layer with the message
# m.get_root().html.add_child(folium.Element(message_html))
#
#
# m.save("mergebeforeweek2day10withTreshold.html")