from nextbike.constants import CONSTANTS

import os
import warnings
warnings.filterwarnings('ignore')

from..utils import cast_address_to_coord

try:
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import geopy

    import folium
    from folium import plugins
    from folium.plugins import HeatMapWithTime
    from folium.plugins import MarkerCluster
    from folium import plugins
    from folium.plugins import HeatMap

    from shapely.geometry import Point
    from sklearn.utils import shuffle

    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt

except ImportError as e:
    print("Import Error...", e)


def __read_geo_data(gson):
    try:
        df = gpd.read_file(os.path.join(CONSTANTS.PATH_EXTERNAL, gson))
        return df
    except ImportWarning as e:
        print("Dataframe ", e)


def __prep_geo_data(df):
    # filter for districts of dortmund
    df = df[df["note"].str.contains("Dortmund")]

    # calculate the center of the districts (for later analysis)
    df["longitude"] = df["geometry"].centroid.x
    df["latitude"] = df["geometry"].centroid.y


def make_point(row):
    return Point(row.longitude_start, row.latitude_start)


# ab task c
def create_map(shape="dortmund_plz.geojson", center=CONSTANTS.CENTER_OF_DORTMUND, tiles=CONSTANTS.TILES,
               attr=CONSTANTS.ATTR, zoom_start=12, min_zoom=11, height="100%", width="100%"):

    map = folium.Map(
        attr=attr,
        location=center,
        tiles=tiles,
        zoom_start=zoom_start,
        min_zoom=min_zoom,
        height=height,
        width=width
    )
    folium.GeoJson(__read_geo_data(shape), name='geojson').add_to(map)

    return map


def station_capacity(df):
    tmp_map = create_map()
    tmp_map.add_child(plugins.HeatMap(df["coordinates_start"], radius=20))
    tmp_map.add_child(plugins.HeatMap(df["coordinates_end"], radius=20))

    return tmp_map


def most_used_station(df, amount=1000):
    tmp_map = create_map()
    mc = MarkerCluster()
    df = shuffle(df)

    i = 0
    for index, row in df.iterrows():
        if i <= amount:
            mc.add_child(folium.Marker(location=[row["latitude_start"], row["longitude_start"]]))
            mc.add_child(folium.Marker(location=[row["latitude_end"], row["longitude_end"]]))
            tmp_map.add_child(mc)
            i = i + 1
        else:
            break
    return tmp_map


def show_trips(df, amount=500):
    tmp_map = create_map()

    df = shuffle(df)

    i = 0
    for index, row in df.iterrows():
        if i <= amount:
            folium.ColorLine([row["coordinates_start"], row["coordinates_end"]],
                             colors=[0, 1, 2],
                             colormap=["blue", "green"],
                             weight=1,
                             opacity=0.3).add_to(tmp_map)
            i = i + 1
        else:
            break
    return tmp_map


def show_map_at_specific_day(df, date="2019-01-20", street="Signal Iduna Park", coord=[]):
    tmp_map = create_map()

    if not coord:
        loc = cast_address_to_coord(street)
    else:
        loc = {"latitude": coord[0],
               "longitude": coord[1]}

    folium.Marker(location=[loc.latitude, loc.longitude],
                  popup=loc,
                  icon=folium.Icon(color='blue'),
                  ).add_to(tmp_map)

    df_tmp = df[(df['datetime_start'] >= date + " 00:00:00") & (df['datetime_start'] <= date + " 23:59:59")]

    tmp_map.add_child(plugins.HeatMap(df_tmp["coordinates_start"], radius=20))
    tmp_map.add_child(plugins.HeatMap(df_tmp["coordinates_end"], radius=20))

    for index, row in df_tmp.iterrows():
        folium.ColorLine([row["coordinates_start"], row["coordinates_end"]],
                         colors=[0, 1, 2],
                         colormap=["red", "blue"],
                         weight=1,
                         opacity=0.5).add_to(tmp_map)

    return tmp_map

