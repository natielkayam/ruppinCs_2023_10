
import numpy as np
import pandas as pd
import geopandas as gpd



Alerts_df = pd.read_csv('DataWazeAlerts.csv',low_memory=False)
Alerts_df.columns = ['TS', 'AlertID', 'PubDate', 'GeoRSS', 'Linqmap_Uuid', 'Linqmap_Magvar', 'Linqmap_Type',
                     'Linqmap_SubType', 'Linqmap_ReportDescription', 'Linqmap_Street', 'Linqmap_City',
                     'Linqmap_Country', 'Linqmap_RoadType', 'Linqmap_ReportRating', 'Linqmap_JamUuid',
                     'Linqmap_Reliability', 'Linqmap_ReportByMunicipalityUser', 'Linqmap_ThumbsUp',
                     'Linqmap_Confidence', 'Waze_geom']
#keep only prmary street
Alerts_df.query("Linqmap_RoadType == 2", inplace=True)
# print(Alerts_df['Linqmap_SubType'].unique())
# print(Alerts_df['Linqmap_SubType'].nunique())
# print(Alerts_df['Linqmap_SubType'].isna().sum())
# print(len(Alerts_df[Alerts_df['Linqmap_SubType'] == 'NO_SUBTYPE']))

angle_dict = {range(338, 361): "N",
              range(0, 23): "N",
              range(23, 68): "NE",
              range(68, 113): "E",
              range(113, 158): "SE",
              range(158, 203): "S",
              range(203, 248): "SW",
              range(248, 293): "W",
              range(293, 338): "NW"}

Alerts_df["Angle"] = Alerts_df["Linqmap_Magvar"].apply(lambda x: next((v for k, v in angle_dict.items() if x in k), ""))

#check if there is group with same uuid and pubdate and geo = 69004
groups = Alerts_df.groupby(["Linqmap_Uuid","PubDate"])



#todo: merge every group to 1 row -
#1) count the numbers of the rows before and add column with count
#2) difference in relability - make max and min column
#3) duration is = the last TS - pubdate

# Group the dataframe
groups = Alerts_df.groupby(["Linqmap_Uuid","PubDate","GeoRSS", "Angle"])
# Merge the rows in each group
merged = groups.apply(lambda x: x.fillna(method='ffill').iloc[-1])
# Add the count column
merged["Count"] = groups.size().values
# Add the max and min_max_reliability columns
merged["Max_Reliability"] = groups["Linqmap_Reliability"].max().values
merged["Min_Reliability"] = groups["Linqmap_Reliability"].min().values
# Add the latest TS column for each group
merged["TS_latest"] = groups.apply(lambda x: x.sort_values("TS", ascending=False).head(1)["TS"].values[0])
# Convert PubDate and TS to datetime
merged["PubDate"] = pd.to_datetime(merged["PubDate"])
merged["TS"] = pd.to_datetime(merged["TS"])
merged["TS_latest"] = pd.to_datetime(merged["TS_latest"])
merged["Duration_Alerts"] = (merged['TS_latest'] - merged['PubDate'])
merged["TS"] = merged["TS_latest"]
merged.drop(columns= 'TS_latest', inplace=True)

Alerts_df = merged
Alerts_df.reset_index(drop=True, inplace=True)
Alerts_df['week'] = Alerts_df['PubDate'].dt.isocalendar().week

Alerts_df[['Lat', 'long']] = Alerts_df.GeoRSS.str.split(" ", expand=True)
new_df = gpd.GeoDataFrame(
    Alerts_df,
    geometry=gpd.points_from_xy(Alerts_df.Lat, Alerts_df.long), crs="EPSG:4326")
Alerts_df.drop(
    columns=["Linqmap_Type", "Linqmap_JamUuid", "Linqmap_ReportDescription", "Linqmap_City", "Linqmap_Country",
             "Linqmap_RoadType", "Linqmap_ReportByMunicipalityUser", "Linqmap_ThumbsUp", "Waze_geom","AlertID","TS","GeoRSS", "Lat", "long"],
    inplace=True)

Alerts_df.to_csv("Alert_2304.csv")

