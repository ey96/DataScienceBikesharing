# these are currently the coordinates of TU Dortmund Hörsaalgebäude 2
from vincenty import vincenty
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import geopandas as gpd
import numpy as np
import warnings


def prepare_data(df):
    """

    :param df:
    :return:
    """
    # ToDo: Diesen geometry-step könnte man auch bei preprocessing reinpacken, dann müssen wir das nicht doppelt berechnen

    # Go through every row, and make a point out of its lat and lon
    df["geometry"] = df.apply(make_point, axis=1)
    # It doesn't come with a CRS because it's a CSV, so it has to be set
    df.crs = {'init': 'epsg:4326'}

    # get geodata of germany (postal codes and their areas/polygons)
    districts_germany = gpd.read_file("../data/external/germany_postalcodes.geojson")
    # filter for districts of dortmund
    districts_dortmund = districts_germany[districts_germany["note"].str.contains("Dortmund")]

    # convert dataset of trips to geodataframe (so it can be merged later with the geodataframe of dortmund)
    geo_df = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=df.geometry)

    # join the data
    # merges data when POINT of trips is within POLYGON of a dortmund district
    df_merged = gpd.sjoin(geo_df, districts_dortmund, how='left', op='within')

    # adding the distance between start position and the center of the university
    df_merged["distanceToUniversity"] = df_merged.apply(calculate_distance_to_university, axis=1)

    # add the attribute whether a trip was done towars/away from university
    university_stations = ["TU Dortmund Seminarraumgebäude 1", "TU Dortmund Hörsaalgebäude 2", "Universität/S-Bahnhof",
                           "TU Dortmund Emil-Figge-Straße 50", "FH-Dortmund Emil-Figge-Straße 42"]

    df_merged['towardsUniversity'] = df_merged['p_name_end'].apply(lambda x: 1 if x in university_stations else 0)
    df_merged['awayFromUniversity'] = df_merged['p_name_start'].apply(lambda x: 1 if x in university_stations else 0)

    df_merged['tripLabel'] = df_merged.apply(lambda row: __get_trip_label(row), axis=1)

    # df_merged['area_start'] = ... # if we receive input data we have to calculate the attribute 'area_start'
    # here I don't do it, because it already exists

    return df_merged


def calculate_distance_to_university(row):
    """

    :param row:
    :return:
    """
    # mean of the university-station-coordinates
    university_center_lat = (51.492296 + 51.491721 + 51.49269 + 51.493966 + 51.493695) / 5
    university_center_lon = (7.41273 + 7.409468 + 7.417633 + 7.418008 + 7.420396) / 5

    # distance of the start station to the "center" of the university-area
    distance = vincenty([row["latitude_start"], row["longitude_start"]],
                        [university_center_lat, university_center_lon], )

    return distance


def __get_trip_label(row):
    """

    :param row:
    :return:
    """
    if (row['towardsUniversity'] == 1) & (row['awayFromUniversity'] == 0):
        return 'towardsUniversity'
    if (row['towardsUniversity'] == 0) & (row['awayFromUniversity'] == 1):
        return 'awayFromUniversity'
    if (row['towardsUniversity'] == 1) & (row['awayFromUniversity'] == 1):
        return 'towardsUniversity'
    if (row['towardsUniversity'] == 0) & (row['awayFromUniversity'] == 0):
        return 'noUniversityRide'

    warnings.warn("Warning...........Message")
    return None


def make_point(row):
    return Point(row.longitude_start, row.latitude_start)


def __get_result(y_test, y_pred, y_train, y_pred_train):
    print("w/o cross-validaiton:")
    print("R2-Score is: {}".format(r2_score(y_train,y_pred_train)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train,y_pred_train))))
    print("MAE: {}".format(mean_absolute_error(y_train,y_pred_train)))
    print("")
    print("w/ cross-validation")
    print("R2-Score is: {}".format(r2_score(y_test,y_pred)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,y_pred))))
    print("MAE: {}".format(mean_absolute_error(y_test,y_pred)))

