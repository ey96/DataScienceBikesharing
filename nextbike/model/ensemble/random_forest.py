from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from nextbike.io.output import __save_model, __save_prediction
from nextbike.io.input import __read_model, read_file
from nextbike.constants import CONSTANTS

import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

# arrays for results
model, r2, rmse, mae, exetime, desc = [], [], [], [], [], []

dic = {
    'name': model,
    'r2': r2,
    'rmse': rmse,
    "mae": mae,
    "exetime": exetime,
    "desc": desc
}


def __init__(df, log=False):
    # best features so far
    X = df[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start",
            "area_start", "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]

    if log is True:
        df["log_duration"] = df["trip_duration"].transform(np.log)
        y = df["log_duration"]
    else:
        y = df["trip_duration"]

    # split data / cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }


def rfr(init, estimator=RandomForestRegressor()):
    # first try rfr
    start = time.time()  # to measure execution time

    # fitting a random forrest regressor
    rfr = estimator
    rfr.fit(init['X_train_scaled'], init['y_train'])
    pred = rfr.predict(init['X_test_scaled'])
    pred_train = rfr.predict(init['X_train_scaled'])

    end = time.time()
    execution_time = (end - start) / 6

    _get_results(rfr, init['X_train_scaled'], pred_train, init['X_test_scaled'], pred, init)

    model.append("RF")
    r2.append(rfr.score(init['X_test_scaled'], init['y_test']))
    rmse.append(np.sqrt(mean_squared_error(init['y_test'], pred)))
    mae.append(mean_absolute_error(init['y_test'], pred))
    exetime.append(execution_time)  # save execution time from the cell above
    desc.append("first try rfr, overfitted ")  # describe the model above

    return {
        'pred': pred,
        'pred_train': pred_train
    }


# best model
def train(init):
    start = time.time()  # to measure execution time

    #print('init training parameters')
    # fitting a random forrest regressor
    rfr = RandomForestRegressor(max_features="auto", n_estimators=1155, max_depth=70, min_samples_split=10,
                                min_samples_leaf=8, bootstrap=True)

    #print('training started')
    rfr.fit(init['X_train_scaled'], init['y_train'])
    #print('training done')

    __save_model(rfr, 'duration_model')
    print('model saved under data/output/duration_model.pkl')

    pred = rfr.predict(init['X_test_scaled'])
    pred_train = rfr.predict(init['X_train_scaled'])

    _get_results(rfr, init['X_train_scaled'], pred_train, init['X_test_scaled'], pred, init)
    end = time.time()
    execution_time = (end - start) / 60

    print('execution time:', execution_time)


def predict(df_trips, df_test):

    X_train = df_trips[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start",
                        "area_start", "temperature °C", "precipitation", "distanceToUniversity",
                        "distanceToCentralStation"]]

    y_train = df_trips["trip_duration"]

    X_test = df_test[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start",
                      "area_start", "temperature °C", "precipitation", "distanceToUniversity",
                      "distanceToCentralStation"]]

    y_test = df_test["trip_duration"]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    mod = __read_model('duration_model')
    print('trained model successfully imported')

    pred = mod.predict(X_test_scaled)
    pred_train = mod.predict(X_train_scaled)

    df_pred = pd.DataFrame(pred, columns=['predictions'])
    __save_prediction(df_pred, 'duration_prediction')
    print('predict values saved under data/output/duration_prediction.csv')

    print("w/o cross-validation:")
    print("R^2-Score is: {}".format(mod.score(X_train_scaled, y_train)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train, pred_train))))
    print("MAE: {}".format(mean_absolute_error(y_train, pred_train)))
    print("")
    print("w/ cross-validation")
    print("R2-Score is: {}".format(mod.score(X_test_scaled, y_test)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, pred))))
    print("MAE: {}".format(mean_absolute_error(y_test, pred)))


def _get_results(model, X_train_scaled,pred_train,X_test_scaled, pred, init):
    print("w/o cross-validation:")
    print("R^2-Score is: {}".format(model.score(X_train_scaled, init['y_train'])))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_train'], pred_train))))
    print("MAE: {}".format(mean_absolute_error(init['y_train'], pred_train)))
    print("")
    print("w/ cross-validation")
    print("R2-Score is: {}".format(model.score(X_test_scaled, init['y_test'])))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_test'], pred))))
    print("MAE: {}".format(mean_absolute_error(init['y_test'], pred)))


def convert_log_to_exp(init, rfr):
    # summary of results
    y_train = np.exp(init['y_train'])
    pred_train = np.exp(rfr['pred_train'])

    y_test = np.exp(init['y_test'])
    pred = np.exp(rfr['pred'])

    print("w/o cross-validation:")
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train, pred_train))))
    print("MAE: {}".format(mean_absolute_error(y_train, pred_train)))
    print("")
    print("w/ cross-validation")
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, pred))))
    print("MAE: {}".format(mean_absolute_error(y_test, pred)))

    model.append("RF")
    r2.append("-")
    rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    mae.append(mean_absolute_error(y_test, pred))
    exetime.append(0)
    desc.append("using log of trip duration/ exp after prediction")


'''
def optimize_hyper_parameters_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # random forest model creation
    rfc = RandomForestClassifier()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)
    # Fit the model
    rfc_random.fit(X_train, y_train)
    # print results
    print(rfc_random.best_params_)
'''