import os
from vincenty import vincenty
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import warnings


def __read_file(path=None):
    """
    :param path: Path of the source file, if path = None dortmund.csv will be used.
    :return: Read data as DataFrame
    """

    if path is None:
        path = os.path.join(os.getcwd(), 'data/input/dortmund.csv')
    try:
        df = pd.read_csv(path,index_col=0)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def __isWeekend(index_of_day):
    """
    :param index_of_day: Weekday in integers (e.g. Monday = 0, Sunday = 6)
    :return: True for Saturday (index_of_day = 5) and Sunday (index_of_day = 5), otherwise False
    """
    if index_of_day > 4:
        return True
    else:
        return False


def __addFeatureColumns(df_final=None):
    """"
    :param df_final: Dataframe, that should be extended with new feature columns.
    """

    # adding the trip duration with the difference of start and end time
    df_final["trip_duration"] = df_final["datetime_end"] - df_final["datetime_start"]

    # converting timedelta to numeric and format in minutes
    df_final["trip_duration"] = pd.to_numeric(df_final["trip_duration"] / 60000000000)

    # adding the distance between start and end position
    df_final["distance"] = df_final.apply(
        lambda x: vincenty([x["latitude_start"], x["longitude_start"]],
                           [x["latitude_end"], x["longitude_end"]], ), axis=1)

    ## adding the weekday of the start time of a trip; stored in integers (0: monday, 6:sunday)
    df_final['weekday'] = df_final['datetime_start'].dt.dayofweek

    # adding new boolean column "weekend"
    df_final["weekend"] = df_final["weekday"].apply(lambda x: __isWeekend(x))

    # transform column "datatime_start" into several columns
    df_final["day"] = df_final["datetime_start"].apply(lambda x: x.day)
    df_final["month"] = df_final["datetime_start"].apply(lambda x: x.month)
    df_final["hour"] = df_final["datetime_start"].apply(lambda x: x.hour)


def get_trip_data(path=None):
    """
    Reads the csv file and transforms the location data of bikes into trip data.
    :parameter
        path: Directory of the csv file, that should read in.
            Default is None --> reads dortmund.csv
    :return:
        Final Dataframe with added features.
    """

    warnings.filterwarnings('ignore')
    df = __read_file(path)
    df_start = df[(df["trip"] == "start") & (df["p_number"] != 0)]
    df_end = df[(df["trip"] == "end") & (df["p_number"] != 0)]
    df_start.reset_index(inplace=True)
    df_end.reset_index(inplace=True)

    # rename columns for merging
    df_start.rename(columns={"index": "index_start", "datetime": "datetime_start", "p_lat": "latitude_start",
                             "p_lng": "longitude_start", "p_name": "p_name_start", "b_number": "b_number_start"},
                    inplace=True)
    df_end.rename(
        columns={"index": "index_end", "datetime": "datetime_end", "p_lat": "latitude_end", "p_lng": "longitude_end",
                 "p_name": "p_name_end", "b_number": "b_number_end"}, inplace=True)

    # drop redundant columns
    df_start.drop(['p_spot', 'p_place_type', 'trip',
                   'p_uid', 'p_bikes', 'b_bike_type',
                   'p_number', 'p_bike'], inplace=True, axis=1)

    df_end.drop(['p_spot', 'p_place_type', 'trip',
                 'p_uid', 'p_bikes', 'b_bike_type',
                 'p_number', 'p_bike'], inplace=True, axis=1)

    # modify the index_end to merge the dataframes by index_start and index_end
    df_end["index_end"] = df_end["index_end"] - 1

    # merge the two sepearte dataframes to the final dataframe
    # the final dataframe consists of datasets which describe a trip with features for the start and the end of a trip
    df_final = pd.merge(df_start, df_end, left_on="index_start", right_on="index_end")

    # drop the redundant columns
    df_final.drop(["index_start", "index_end", "b_number_end"], inplace=True, axis=1)
    df_final.rename({"b_number_start": "b_number"}, inplace=True)

    # converting objects to datetimes
    df_final["datetime_start"] = pd.to_datetime(df_final["datetime_start"])
    df_final["datetime_end"] = pd.to_datetime(df_final["datetime_end"])

    __addFeatureColumns(df_final=df_final)

    return df_final


def get_write_trip_data():
    """
    Transforms the data to trip data and saves the final dataframe in a new csv file.
    """

    pd.DataFrame(data=get_trip_data()).to_csv('data/output/dortmund_trips.csv')
    print("Transformed trip data for Dortmund successfully saved in a csv file!")
