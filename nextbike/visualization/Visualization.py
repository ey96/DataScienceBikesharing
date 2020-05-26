from nextbike.constants import CONSTANTS

import os
import warnings
warnings.filterwarnings('ignore')

from ..utils import cast_address_to_coord
from ..io import input

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


# ab task c
def create_map(shape="dortmund_plz.geojson", center=CONSTANTS.CENTER_OF_DORTMUND.value, tiles=CONSTANTS.TILES.value,
               attr=CONSTANTS.ATTR.value, zoom_start=12, min_zoom=11, height="100%", width="100%"):
    """
    Creats a Folium-Map with dortmund as a shape.

    :param shape: name of the .geojson-file, which is stored under data/external
    :param center: specify your custom center default: [51.511838, 7.456943]
    :param tiles:  specify your custom tiles default: cartodbpositron
    :param attr:   specify your custom attr
    :param zoom_start: specify the zoom_start of the map default is: 12
    :param min_zoom: specify the min_zoom of the map default is: 11
    :param height: specify the height of the map default is: 100%
    :param width: specify the width of the map default is: 100%
    :return: a Folium-map with the your either your preferences or the default settings
    """
    map = folium.Map(
        attr=attr,
        location=center,
        tiles=tiles,
        zoom_start=zoom_start,
        min_zoom=min_zoom,
        height=height,
        width=width
    )
    folium.GeoJson(input.__read_geojson(shape), name='geojson').add_to(map)

    return map


def show_station_map(df):

    """

    :param df:
    :return:
    """

    stations_map = create_map()

    for index, row in df.iterrows():
        station_rentals = row['count']

        station_name = row['p_name_start']

        station_info = "Name: {}\n\nAmount of rentals: {}\n".format(station_name, station_rentals)

        folium.Circle(
            location=[row['latitude_start'], row['longitude_start']],
            popup=station_info,
            radius=row['count'] * 0.009,
            color='red',
            fill=True,
            fill_color='#red'
        ).add_to(stations_map)

    return stations_map


def show_rental_for_june(df):
    """

    :param df:
    :return:
    """

    df["count"] = df["count"].fillna(0)

    quan = np.arange(0, 1.1, 0.1)
    bins = list(df['count'].quantile(quan))
    districts_map = create_map(zoom_start=10.5)

    folium.Choropleth(
        geo_data=df,
        data=df,
        columns=['plz', 'count'],
        key_on='properties.plz',
        fill_color='RdYlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        line_weight=1,
        line_color='black',
        legend_name='Number of bookings per postcode',
        bins=bins
    ).add_to(districts_map)

    # Show information (postal code and rental amount) when mouse over district
    folium.GeoJson(
        df,
        style_function=lambda feature: {"color": "black", "weight": 1, },
        highlight_function=lambda x: {"weight": 2, "color": "black", "fillOpacity": 0.5},
        tooltip=folium.features.GeoJsonTooltip(fields=['count', 'plz'], aliases=['Amount of rentals:', 'Postcal code:'])
    ).add_to(districts_map)

    return districts_map


def show_time_heatmap(df, df2):
    """

    :param df:
    :param df2:
    :return:
    """
    # visualize
    heatmap_daily = create_map(zoom_start=11.2)

    plugins.HeatMapWithTime(df,
                            index=['5:00 - 9:00', '10:00 - 15:00', '16:00 - 20:00', '21:00 - 4:00'],
                            auto_play=True,
                            radius=30,
                            overlay=False,
                            use_local_extrema=True).add_to(heatmap_daily)

    folium.Choropleth(
        geo_data=df2,
        fill_opacity=0.1,
        line_opacity=1, ).add_to(heatmap_daily)

    return heatmap_daily


def show_heatmap_monthly_per_district(df, df2):

    heatmap_monthly_per_district = create_map(zoom_start=11.2)

    plugins.HeatMapWithTime(df,
                            index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            auto_play=True,
                            radius=30,
                            overlay=False,
                            use_local_extrema=True).add_to(heatmap_monthly_per_district)

    folium.Choropleth(
        geo_data=df2,
        fill_opacity=0.1,
        line_opacity=1, ).add_to(heatmap_monthly_per_district)

    return heatmap_monthly_per_district


def station_capacity(df, radius=20):
    """
    Creat's a Folium-Heatmap based on the start-coordinates and end-coordinates

    :param df: the dataframe
    :param radius: specify the radius of the heatmap default: 20
    :return: Folium-Heatmap
    """
    tmp_map = create_map()
    tmp_map.add_child(plugins.HeatMap(df[["latitude_start", "longitude_start"]], radius=radius))
    tmp_map.add_child(plugins.HeatMap(df[["latitude_end", "longitude_end"]], radius=radius))

    return tmp_map


def most_used_station(df, random=True, amount=1000):
    """
    Creats a grouping of the most-used-stations. You can either choose to picks random stations or the first "amount"
     of stations in the df

    :param df: the dataframe
    :param random: Pick random rows. Or you can set random to False to get the first "amount" of stations in the df
    default: True
    :param amount: how much random rows should be included
    :return: Folium-Map
    """
    tmp_map = create_map()
    mc = MarkerCluster()
    if random:
        df = shuffle(df)

    i = 0
    for index, row in df.iterrows():
        if i < amount:
            mc.add_child(folium.Marker(location=[row["latitude_start"], row["longitude_start"]]))
            mc.add_child(folium.Marker(location=[row["latitude_end"], row["longitude_end"]]))
            tmp_map.add_child(mc)
            i = i + 1
        else:
            break
    return tmp_map


def show_trips(df, random=True, amount=500):
    """
    Creat's a Foliom-ColorLine, which shows random trips. You can either choose to picks random trips or the
    first "amount" of trips in the df

    :param df: the dataframe
    :param random: Pick random rows. Or you can set random to False to get the first "amount" of trips in the df
    default: True
    :param amount: The amount of trips, you want to show (ATTENTION: needs a lot of RAM)
    :return: Folium-ColorLine Map
    """
    tmp_map = create_map()
    if random:
        df = shuffle(df)

    i = 0
    for index, row in df.iterrows():
        if i <= amount:
            folium.ColorLine([[row["latitude_start"],row["longitude_start"]],[row["latitude_end"],row["longitude_end"]]],
                             colors=[0, 1, 2],
                             colormap=["blue", "green"],
                             weight=1,
                             opacity=0.3).add_to(tmp_map)
            i = i + 1
        else:
            break
    return tmp_map


# muss noch ausgearbeitet werden
def show_map_at_specific_day(df, date="2019-01-20", street="Signal Iduna Park", coord=[]):
    """

    :param df:
    :param date:
    :param street:
    :param coord:
    :return:
    """
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

