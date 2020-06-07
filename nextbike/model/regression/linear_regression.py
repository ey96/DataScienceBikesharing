from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from nextbike.model.regression.parameters import model_dic
from nextbike.io.output import save_model

import time
import numpy as np
import pandas as pd


def __init__(df):

    X = df[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start", "area_start",
            "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def linear_regression(init):
    # fit simple linear regression

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(init['X_train'])
    X_train_scaled = scaler.transform(init['X_train'])

    scaler.fit(init['X_test'])
    X_test_scaled = scaler.transform(init['X_test'])

    mod = LinearRegression()
    mod.fit(X_train_scaled, init['y_train'])

    #save_model(mod, 'LinearRregression')
    #print('Model saved')

    y_pred = mod.predict(X_test_scaled)
    y_pred_train = mod.predict(X_train_scaled)

    get_result(init['y_test'], y_pred, init['y_train'], y_pred_train)


def lasso_regression(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    mod = Lasso()
    mod.fit(X_train_scaled,y_train,)

    y_pred = mod.predict(X_test_scaled)
    y_pred_train = mod.predict(X_train_scaled)

    get_result(y_test, y_pred, y_train, y_pred_train)


def ridge_regression(X_train, X_test, y_train, y_test):
    # Ridge regression
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    mod = Ridge(fit_intercept=True)
    mod.fit(X_train_scaled, y_train)

    y_pred = mod.predict(X_test_scaled)
    y_pred_train = mod.predict(X_train_scaled)

    get_result(y_test, y_pred, y_train, y_pred_train)


def compare_linear_regression_models(X_train, X_test, y_train, y_test):
    models = [LinearRegression(), Lasso(), Ridge()]
    names = ["Linear", "Lasso", "Ridge"]
    RMSE = []
    R2 = []
    MAE = []
    exetime = []
    desc = []

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    for i in range(len(models)):
        estimator = models[i]

        start = time.time()

        estimator.fit(X_train_scaled, y_train)
        y_pred = estimator.predict(X_test_scaled)

        end = time.time()

        RMSE.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        R2.append(r2_score(y_test, y_pred))
        MAE.append(mean_absolute_error(y_test, y_pred))
        exetime.append((end - start))
        desc.append("")

    result_dict = {"Algorithm": names,
                   "RMSE": RMSE,
                   "R2": R2,
                   "MAE": MAE,
                   "Execution time (sec)": exetime,
                   "Description": desc}

    df_result = pd.DataFrame(result_dict)
    return df_result


def calculate_hyper_parameters(X_train, X_test, y_train):
    # GridSearch to find optimal hyperparameters
    scores = []

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)

    for model_name, model_param in model_dic.items():
        estimator = GridSearchCV(model_param["model"], model_param["parameters"], cv=5, return_train_score=False)
        estimator.fit(X_train_scaled, y_train)
        scores.append({"model": model_name,
                       "best_score": estimator.best_score_,
                       "alpha": estimator.best_params_["alpha"],
                       "max_iter": estimator.best_params_["max_iter"],
                       "random_state": estimator.best_params_["random_state"],
                       "fit_intercept": estimator.best_params_["fit_intercept"],
                       })

    df_results = pd.DataFrame(scores)
    return df_results


def get_result(y_test, y_pred, y_train, y_pred_train):
    print("w/o cross-validaiton:")
    print("R2-Score is: {}".format(r2_score(y_train,y_pred_train)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train,y_pred_train))))
    print("MAE: {}".format(mean_absolute_error(y_train,y_pred_train)))
    print("")
    print("w/ cross-validation")
    print("R2-Score is: {}".format(r2_score(y_test,y_pred)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,y_pred))))
    print("MAE: {}".format(mean_absolute_error(y_test,y_pred)))