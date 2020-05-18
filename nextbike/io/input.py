import pandas as pd
import os
import pickle
import geopandas as gpd
from nextbike.constants import CONSTANTS

from . import utils


def read_gson(gson):
    path = os.path.join(CONSTANTS.PATH_EXTERNAL, gson)
    try:
        df = gpd.read_file(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_csv(loc, name):
    path = os.path.join((CONSTANTS.PATH_RAW if loc == "internal" else CONSTANTS.PATH_PROCESSED), name)
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def __read_file(path=None):
    """
    :param path: Path of the source file, if path = None dortmund.csv will be used.
    :return: Read data as DataFrame
    """
    if path is None:
        path = os.path.join(os.getcwd(), 'data/internal/dortmund.csv')
    try:
        df = pd.read_csv(path, index_col=0)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model():
    path = os.path.join(utils.get_data_path(), "output/model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
