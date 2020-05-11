from nextbike.constants import CONSTANTS

import os
import warnings
warnings.filterwarnings('ignore')

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


# task c
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