import pandas as pd
import os
import pickle
import geopandas as gpd
from nextbike.constants import CONSTANTS

from nextbike import utils


def __read_geojson(geojson):
    """
    Method is private. It reads geojson-files located in the external folder

    :param geojson: name of the .geojson-file, which is located in data/external/
    :return: df of the .geojson
    """
    path = os.path.join(CONSTANTS.PATH_EXTERNAL.value, geojson)
    try:
        df = gpd.read_file(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_csv(loc, name, **kwargs):
    """
    :param loc: internal, processed or external
    :param name: name of the csv
    :return: df of the csv
    """
    if loc.lower() not in ["internal", "external", "processed"]:
        raise Exception('loc has to be either internal, external or processed')
    else:
        if loc.lower() == "internal":
            path = os.path.join(CONSTANTS.PATH_RAW.value, name)
        elif loc.lower() == "external":
            path = os.path.join(CONSTANTS.PATH_EXTERNAL.value, name)
        else:
            path = os.path.join(CONSTANTS.PATH_PROCESSED.value, name)
        try:
            df = pd.read_csv(path, **kwargs)
            return df
        except FileNotFoundError:
            print("Data file not found. Path was " + path)


def __read_file(path=None, **kwargs):
    """
    :param path: Path of the source file, if path = None dortmund.csv will be used.
    :return: Read data as DataFrame
    """
    if path is None:
        path = os.path.join(os.getcwd(), CONSTANTS.PATH_RAW.value + "dortmund.csv")
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def __read_model():
    path = os.path.join(utils.get_data_path(), "output/model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
