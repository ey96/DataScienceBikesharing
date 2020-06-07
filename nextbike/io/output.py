import os
from nextbike.constants import *
from nextbike.constants import __FILE__


def write_trip_data(df):
    """
    saves the final dataframe in a new csv file at /processed.
     :parameter
        df: the dataframe to be saved as csv
    """
    path = os.path.join(__FILE__, CONSTANTS.PATH_PROCESSED.value + "dortmund_trips.csv")
    df.to_csv(path)
    print("Transformed trip data for Dortmund successfully saved in a csv file!")


def __save_trip_data(df, output):
    """
    saves the final dataframe in a new csv file.
     :parameter
        df: the dataframe to be saved as csv
        output: the name of the file
    """
    if output is None:
        df.to_csv(os.path.join(CONSTANTS.PATH_OUTPUT.value))
    else:
        df.to_csv(os.path.join(CONSTANTS.PATH_OUTPUT.value, output))
