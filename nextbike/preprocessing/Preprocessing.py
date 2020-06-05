import os
from vincenty import vincenty
import pandas as pd
import warnings
from shapely.geometry import Point

from ..io import input
from ..constants import CONSTANTS


def __isWeekend(index_of_day):
    """
    :param index_of_day: Weekday in integers (e.g. Monday = 0, Sunday = 6)
    :return: 1 for Saturday (index_of_day = 5) and Sunday (index_of_day = 5), otherwise 0
    """
    if index_of_day > 4:
        return 1
    else:
        return 0


def __get_tripLabel(row):
    if ((row['towardsUniversity'] == 1) & (row['awayFromUniversity'] == 0)):
        return 'towardsUniversity'
    if ((row['towardsUniversity'] == 0) & (row['awayFromUniversity'] == 1)):
        return 'awayFromUniveristy'
    if ((row['towardsUniversity'] == 1) & (row['awayFromUniversity'] == 1)):
        return 'towardsUniversity'
    if ((row['towardsUniversity'] == 0) & (row['awayFromUniversity'] == 0)):
        return 'noUniversityRide'

    warnings.warn("Warning...........Message")
    return None


def __addFeatureColumns(df_final=None, df_weather=None):

    """"
    :param df_final: Dataframe, that should be extended with new feature columns.
    """

    # adding the trip duration with the difference of start and end time
    df_final["trip_duration"] = df_final["datetime_end"] - df_final["datetime_start"]

    # converting timedelta to numeric and format in minutes
    df_final["trip_duration"] = pd.to_numeric(df_final["trip_duration"] / 60000000000)

    df_final["coordinates_start"] = list(zip(df_final["latitude_start"],df_final["longitude_start"]))
    df_final["coordinates_end"] = list(zip(df_final["latitude_end"],df_final["longitude_end"]))

    # adding the distance between start and end position
    df_final["distance"] = df_final.apply(
        lambda x: vincenty([x["latitude_start"], x["longitude_start"]],
                           [x["latitude_end"], x["longitude_end"]], ), axis=1)

    # adding another distances
    df_final["distanceToUniversity"] = df_final.apply(lambda x: vincenty([x["latitude_start"], x["longitude_start"]],
                                                                         [51.4928736,7.415647], ), axis=1)
    df_final["distanceToCentralStation"] = df_final.apply(
        lambda x: vincenty([x["latitude_start"], x["longitude_start"]],
                           [51.5175, 7.458889], ), axis=1)

    # adding the weekday of the start time of a trip; stored in integers (0: monday, 6:sunday)
    df_final['weekday'] = df_final['datetime_start'].dt.dayofweek

    # adding new boolean column "weekend"
    df_final["weekend"] = df_final["weekday"].apply(lambda x: __isWeekend(x))

    # transform column "datatime_start" into several columns
    df_final["day"] = df_final["datetime_start"].apply(lambda x: x.day)
    df_final["month"] = df_final["datetime_start"].apply(lambda x: x.month)
    df_final["hour"] = df_final["datetime_start"].apply(lambda x: x.hour)
    df_final["minute"] = df_final["datetime_start"].apply(lambda x: x.minute)
    df_final["day_of_year"] = df_final["datetime_start"].apply(lambda x: x.timetuple().tm_yday)

    # add the attribute whether a trip was done towards/away from university
    #array with university stations
    university_stations = ["TU Dortmund Seminarraumgebäude 1", "TU Dortmund Hörsaalgebäude 2", "Universität/S-Bahnhof",
                           "TU Dortmund Emil-Figge-Straße 50", "FH-Dortmund Emil-Figge-Straße 42"]

    df_final['towardsUniversity'] = df_final['p_name_end'].apply(lambda x: 1 if x in university_stations else 0)
    df_final['awayFromUniversity'] = df_final['p_name_start'].apply(lambda x: 1 if x in university_stations else 0)

    df_final['tripLabel'] = df_final.apply(lambda row: __get_tripLabel(row), axis=1)

    if df_weather is not None:
        df_final["datetime_start_for_merge_with_weather"] = df_final["datetime_start"].apply(
             lambda x: __formatDatetimeForMerging(str(x)))

        # merge with weather data
        df_final = pd.merge(df_final, df_weather, left_on="datetime_start_for_merge_with_weather", right_on="datetime",
                            how='inner')

        # drop redundant columns
        df_final.drop(labels=["datetime", "datetime_start_for_merge_with_weather"], axis=1, inplace=True)

    return df_final


def __formatDatetimeForMerging(x):
    # return as integer for merging
    return int(x[:13].replace('-', '').replace(' ', ''))


def __readWeatherFiles():
    # read weather data
    # temperature for each hour in 2019
    temp = input.__read_file(os.path.join(CONSTANTS.PATH_EXTERNAL.value, "WaltropTemp.txt"), sep=";")
    temp.rename(columns={"TT_TU": "temperature °C", "MESS_DATUM": "datetime"}, inplace=True)
    temp.drop(labels=["STATIONS_ID", "QN_9", "eor", "RF_TU"], axis=1, inplace=True)
    temp = temp[(temp["datetime"] >= 2019010100) & (temp["datetime"] <= 2019123123)]
    temp.reset_index(drop=True, inplace=True)

    # two features (precipitation in mm & precipitaion y/n) for each hour in 2019
    precipitation = input.__read_file(os.path.join(CONSTANTS.PATH_EXTERNAL.value, "WaltropPrecipitation.txt"), sep=";")
    precipitation.rename(columns={"  R1": "precipitation in mm", "MESS_DATUM": "datetime", "RS_IND": "precipitation"},
                         inplace=True)
    precipitation = precipitation[(precipitation["datetime"] >= 2019010100) & (precipitation["datetime"] <= 2019123123)]
    precipitation.drop(labels=["STATIONS_ID", "QN_8", "eor", "WRTR"], axis=1, inplace=True)
    precipitation.reset_index(drop=True, inplace=True)
    weather = pd.merge(temp, precipitation, on="datetime")

    return weather


def get_trip_data(path=None, withWeather=True):
    """
    Reads the csv file and transforms the location data of bikes into trip data.
    :parameter
        path: Directory of the csv file, that should read in.
            Default is None --> reads dortmund.csv
        withWeather: adds weather features to the final DataFrame
    :return:
        Final DataFrame with added features.
    """

    warnings.filterwarnings('ignore')

    df = input.__read_file(path)
    df =df[((df["trip"] == "start") | (df["trip"]=="end"))]

    deletionFilter = df["trip"] != df["trip"].shift(-1)
    df = df[deletionFilter]

    df_start = df[(df["trip"] == "start")]
    df_end = df[(df["trip"] == "end")]


    df_start.reset_index(inplace=True)
    df_end.reset_index(inplace=True)

    # rename columns for merging
    df_start.rename(columns={"index": "index_start", "datetime": "datetime_start", "p_lat": "latitude_start",
                             "p_lng": "longitude_start", "p_name": "p_name_start", "b_number": "b_number_start", "p_number": "p_number_start"},
                    inplace=True)
    df_end.rename(
        columns={"index": "index_end", "datetime": "datetime_end", "p_lat": "latitude_end", "p_lng": "longitude_end",
                 "p_name": "p_name_end", "b_number": "b_number_end","p_number":"p_number_end"}, inplace=True)

    # drop redundant columns
    df_start.drop(['p_spot', 'p_place_type', 'trip',
                   'p_uid', 'p_bikes', 'b_bike_type',
                   'p_bike'], inplace=True, axis=1)

    df_end.drop(['p_spot', 'p_place_type', 'trip',
                 'p_uid', 'p_bikes', 'b_bike_type',
                 'p_bike'], inplace=True, axis=1)

    # modify the index_end to merge the dataframes by index_start and index_end
    df_end["index_end"] = df_end["index_end"] - 1

    # merge the two sepearte dataframes to the final dataframe
    # the final dataframe consists of datasets which describe a trip with features for the start and the end of a trip
    df_final = pd.merge(df_start, df_end, left_on="index_start", right_on="index_end")

    # p_number != 0 --> just focus on the trips from and to an official bike station
    df_final = df_final[(df_final["p_number_start"] != 0) & (df_final["p_number_end"] != 0)]

    # drop the redundant columns
    df_final.drop(["index_start", "index_end", "b_number_end","p_number_start","p_number_end"], inplace=True, axis=1)

    df_final.rename(columns={"b_number_start": "b_number"}, inplace=True)

    # converting objects to datetimes
    df_final["datetime_start"] = pd.to_datetime(df_final["datetime_start"])
    df_final["datetime_end"] = pd.to_datetime(df_final["datetime_end"])




    if withWeather:
        return __addFeatureColumns(df_final=df_final, df_weather= __readWeatherFiles())
    else:
        return __addFeatureColumns(df_final=df_final)


def get_write_trip_data(df):
    """
    saves the final dataframe in a new csv file.
     :parameter
        df: the dataframe to be saved as csv
    """

    df.to_csv(os.path.join(CONSTANTS.PATH_PROCESSED.value, 'dortmund_trips.csv'))
    print("Transformed trip data for Dortmund successfully saved in a csv file!")


def __prep_geo_data(df):
    """
    Filters the Geo-dataframe (postal code areas) of Germany to Dortmund and calculates the center of each postal code area.
    :parameter
        df: DataFrame that contains geographical information (postal code areas) of Germany
    """
    # filter for districts of dortmund
    df = df[df["note"].str.contains("Dortmund")]

    # calculate the center of the districts (for later analysis)
    df["longitude"] = df["geometry"].centroid.x
    df["latitude"] = df["geometry"].centroid.y

    return df


def __make_point(row):
    """
    Transforms the coordinates to geographical points.
    :return:
        Geographical point (numerical)
    """
    return Point(row.longitude_start, row.latitude_start)


def __get_time_data(df):
    # get the trip data for different times of a day
    fife_nine = df.loc[(df.hour < 10) & (df.hour > 5)]
    ten_three = df.loc[(df.hour < 16) & (df.hour > 9)]
    four_eight = df.loc[(df.hour < 21) & (df.hour > 15)]
    nine_four = df.loc[(df.hour < 6) | (df.hour > 20)]

    # aggregate the amount of rentals per station for each time period
    fife_nine = fife_nine.groupby(['latitude_start', 'longitude_start']).count()
    ten_three = ten_three.groupby(['latitude_start', 'longitude_start']).count()
    four_eight = four_eight.groupby(['latitude_start', 'longitude_start']).count()
    nine_four = nine_four.groupby(['latitude_start', 'longitude_start']).count()

    fife_nine.reset_index(inplace=True)
    ten_three.reset_index(inplace=True)
    four_eight.reset_index(inplace=True)
    nine_four.reset_index(inplace=True)

    time_data = [fife_nine[['latitude_start', 'longitude_start', 'hour']].values.tolist(),
                 ten_three[['latitude_start', 'longitude_start', 'hour']].values.tolist(),
                 four_eight[['latitude_start', 'longitude_start', 'hour']].values.tolist(),
                 nine_four[['latitude_start', 'longitude_start', 'hour']].values.tolist()]

    return time_data


def __get_month_data(df):
    # get the data per month
    trips_jan = df.loc[(df.month == 1)]
    trips_feb = df.loc[(df.month == 2)]
    trips_mar = df.loc[(df.month == 3)]
    trips_apr = df.loc[(df.month == 4)]
    trips_may = df.loc[(df.month == 5)]
    trips_jun = df.loc[(df.month == 6)]

    # july does not exist

    trips_aug = df.loc[(df.month == 8)]
    trips_sep = df.loc[(df.month == 9)]
    trips_oct = df.loc[(df.month == 10)]
    trips_nov = df.loc[(df.month == 11)]
    trips_dec = df.loc[(df.month == 12)]

    # aggregate the amount of rentals per (mapped) district
    trips_jan = trips_jan.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_feb = trips_feb.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_mar = trips_mar.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_apr = trips_apr.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_may = trips_may.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_jun = trips_jun.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})

    trips_aug = trips_aug.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_sep = trips_sep.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_oct = trips_oct.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_nov = trips_nov.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})
    trips_dec = trips_dec.groupby("plz").agg({'count': 'sum', 'longitude': 'mean', 'latitude': 'mean'})

    data = [trips_jan[['latitude', 'longitude', 'count']].values.tolist(),
            trips_feb[['latitude', 'longitude', 'count']].values.tolist(),
            trips_mar[['latitude', 'longitude', 'count']].values.tolist(),
            trips_apr[['latitude', 'longitude', 'count']].values.tolist(),
            trips_may[['latitude', 'longitude', 'count']].values.tolist(),
            trips_jun[['latitude', 'longitude', 'count']].values.tolist(),
            trips_aug[['latitude', 'longitude', 'count']].values.tolist(),
            trips_sep[['latitude', 'longitude', 'count']].values.tolist(),
            trips_oct[['latitude', 'longitude', 'count']].values.tolist(),
            trips_nov[['latitude', 'longitude', 'count']].values.tolist(),
            trips_dec[['latitude', 'longitude', 'count']].values.tolist()]

    return data

