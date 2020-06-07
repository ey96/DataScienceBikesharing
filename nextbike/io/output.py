from nextbike.constants import *
from nextbike.constants import __FILE__

import os
import pickle



def write_trip_data(df):
    """
    saves the final dataframe in a new csv file at the following location: 'data/processed/dortmund_trips.csv'
     :parameter
        df: the dataframe to be saved as csv
    """
    path = os.path.join(__FILE__, CONSTANTS.PATH_PROCESSED.value + "dortmund_trips.csv")
    df.to_csv(path)
    print("Transformed trip data for Dortmund successfully saved in a csv file! ")
    print("PATH is " + path)


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


def save_model(model, name):
    """

    :param model:
    :param name:
    :return:
    """
    pickle.dump(model, open(os.path.join(HEAD, os.path.join(__FILE__, CONSTANTS.PATH_OUTPUT.value + name)), 'wb'))


def save_prediction(df, name):
    """

    :param df:
    :param name:
    :return:
    """
    df.to_csv(os.path.join(HEAD, os.path.join(__FILE__, CONSTANTS.PATH_OUTPUT.value + name)), index=False)



