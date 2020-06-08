from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures

import time
import numpy as np

from nextbike.model.regression.parameters import random_grid_poly

# arrays for results
name, poly_degree, r2, rmse, mae, exetime, desc = [], [], [], [], [], [], []
dic = {
    'name': name,
    'poly_degree': poly_degree,
    'r2':r2,
    'rmse': rmse,
    "mae": mae,
    "exetime": exetime,
    "desc": desc
}


def __init__(df):
    X = df[
        ["month", "weekday", "day", "day_of_year", "hour", "minute", "latitude_start", "longitude_start", "area_start",
         "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def polynomial_reg(model_name, estimator, degree, init):
    start = time.time()

    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(init['X_train'])

    model = estimator
    model.fit(x_poly, init['y_train'])

    # evaluate the model on the second set of data
    y_pred = model.predict(poly_reg.transform(init['X_test']))
    y_pred_train = model.predict(x_poly)
    end = time.time()

    __get_result(init, y_pred, y_pred_train)

    name.append(model_name)
    r2.append(r2_score(init['y_test'], y_pred))
    rmse.append(np.sqrt(mean_squared_error(init['y_test'], y_pred)))
    mae.append(mean_absolute_error(init['y_test'], y_pred))
    exetime.append((end - start) / 60)
    desc.append(estimator)
    poly_degree.append(degree)


def __get_result(init, y_pred, y_pred_train):
    print("w/o cross-validation:")
    print("R^2-Score is: {}".format(r2_score(init['y_train'], y_pred_train)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_train'],y_pred_train))))
    print("MAE: {}".format(mean_absolute_error(init['y_train'],y_pred_train)))
    print("")
    print("w/ cross-validation:")
    print ('The R^2 is: ',r2_score(init['y_test'], y_pred))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_test'],y_pred))))
    print("MAE: {}".format(mean_absolute_error(init['y_test'],y_pred)))


def calculate_hyper_parameters(init):
    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(init['X_train'])

    ridge_random = RandomizedSearchCV(estimator=Ridge(), param_distributions=random_grid_poly, n_iter=100,
                                      cv=3, verbose=2, random_state=42, n_jobs=-1)
    # Fit the model
    ridge_random.fit(x_poly, init['y_train'])
    # print results
    print(ridge_random.best_params_)
