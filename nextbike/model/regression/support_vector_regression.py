from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

import time
import numpy as np
from sklearn.svm import SVR

from nextbike.model.utils import __get_result


def __init__(df):
    X = df[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start", "area_start",
            "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }


def train(init, estimator=SVR()):
    start = time.time()
    scaler = StandardScaler()
    scaler.fit(init['X_train'])

    svr = estimator
    svr.fit(init['X_train_scaled'], init['y_train'])

    y_pred = svr.predict(init['X_test_scaled'])
    y_pred_train = svr.predict(init['X_train_scaled'])
    end = time.time()

    __get_result(init['y_test'], y_pred, init['y_train'], y_pred_train)
    # arrays for results
    name, poly_degree, r2, rmse, mae, exetime, desc = [], [], [], [], [], [], []
    name.append("SVR")
    r2.append(r2_score(init['y_test'], y_pred))
    rmse.append(np.sqrt(mean_squared_error(init['y_test'], y_pred)))
    mae.append(mean_absolute_error(init['y_test'], y_pred))
    exetime.append((end - start) / 60)
    desc.append(estimator)