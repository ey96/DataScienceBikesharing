from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from nextbike.model.regression.parameters import model_dic
from nextbike.model.utils import __get_result

import time
import numpy as np
import pandas as pd


def __init__(df):

    X = df[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start", "area_start",
            "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
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


def train(model, init):
    # fit simple regression model w/o tuning

    if model not in ['Lasso', 'Linear', 'Ridge']:
        raise Exception('model has to be either Lasso, Linear or Ridge')

    if model == 'Linear':
        mod = LinearRegression()
    elif model == 'Lasso':
        mod = Lasso()
    else:
        mod = Ridge(fit_intercept=True)

    mod.fit(init['X_train_scaled'], init['y_train'])

    predict(mod, init)


def predict(mod, init):
    """

    :param mod:
    :param init:
    :return:
    """

    y_pred = mod.predict(init['X_test_scaled'])
    y_pred_train = mod.predict(init['X_train_scaled'])

    __get_result(init['y_test'], y_pred, init['y_train'], y_pred_train)


def compare_regression_models(init):
    models = [LinearRegression(), Lasso(), Ridge()]
    names = ["Linear", "Lasso", "Ridge"]
    RMSE = []
    R2 = []
    MAE = []
    exetime = []
    desc = []

    for i in range(len(models)):
        estimator = models[i]

        start = time.time()

        estimator.fit(init['X_train_scaled'], init['y_train'])
        y_pred = estimator.predict(init['X_test_scaled'])

        end = time.time()

        RMSE.append(np.sqrt(mean_squared_error(init['y_test'], y_pred)))
        R2.append(r2_score(init['y_test'], y_pred))
        MAE.append(mean_absolute_error(init['y_test'], y_pred))
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


def calculate_hyper_parameters(init):
    # GridSearch to find optimal hyper-parameters
    scores = []

    for model_name, model_param in model_dic.items():
        estimator = GridSearchCV(model_param["model"], model_param["parameters"], cv=5, return_train_score=False)
        estimator.fit(init['X_train_scaled'], init['y_train'])
        scores.append({"model": model_name,
                       "best_score": estimator.best_score_,
                       "alpha": estimator.best_params_["alpha"],
                       "max_iter": estimator.best_params_["max_iter"],
                       "random_state": estimator.best_params_["random_state"],
                       "fit_intercept": estimator.best_params_["fit_intercept"],
                       })

    df_results = pd.DataFrame(scores)
    return df_results


def model(model, alpha, max_iter, fit_intercept, random_sate, init):
    if model not in["Ridge", "Lasso"]:
        raise Exception('model has to be either Lasso or Ridge')

    start = time.time()  # to measure the execution time
    if model == "Ridge":
        mod = Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept, random_state=random_sate)
        name = "Ridge"
    else:
        mod = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept, random_state=random_sate)
        name = "Lasso"
    mod.fit(init['X_train_scaled'], init['y_train'])
    y_pred = mod.predict(init['X_test_scaled'])
    end = time.time()  # to measure the execution time
    return {
        'names': name,
        'RMSE': np.sqrt(mean_squared_error(init['y_test'], y_pred)),
        'R2': r2_score(init['y_test'], y_pred),
        'MAE': mean_absolute_error(init['y_test'],y_pred),
        'exetime': (end-start),
        'desc': 'Hyperparameters set after GridSearch'

    }



