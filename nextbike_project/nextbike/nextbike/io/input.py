import pandas as pd
import os
import pickle
import numpy as np


def read_file(path=os.path.join(os.getcwd(), 'data/input/dortmund.csv')):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model():
    path = os.path.join(os.getcwd(), 'data/output/model.pkl')
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
