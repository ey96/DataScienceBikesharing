from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import time


def __init__(df):
    # best features so far
    X = df[["month", "weekday", "day_of_year", "hour", "minute", "latitude_start", "longitude_start",
            "area_start", "temperature °C", "precipitation", "distanceToUniversity", "distanceToCentralStation"]]

    y = df["trip_duration"]
    # split data / cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def train(init):
    start = time.time()  # to measure execution time

    # scaling the data
    scaler = StandardScaler()
    scaler.fit(init['X_train'])
    X_train_scaled = scaler.transform(init['X_train'])

    scaler.fit(init['X_test'])
    X_test_scaled = scaler.transform(init['X_test'])

    # fitting a random forrest regressor
    rfr = RandomForestRegressor(max_features="auto", n_estimators=1155, max_depth=70, min_samples_split=10,
                                min_samples_leaf=8, bootstrap=True)
    print('init training parameters')
    rfr.fit(X_train_scaled, init['y_train'])
    pred = rfr.predict(X_test_scaled)
    pred_train = rfr.predict(X_train_scaled)
    print('Training done')

    _get_results(rfr, X_train_scaled, pred_train, X_test_scaled, pred, init)
    end = time.time()
    execution_time = (end - start) / 60

    print('execution time:', execution_time)


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


def _get_results(model,X_train_scaled,pred_train,X_test_scaled, pred, init):
    print("w/o cross-validation:")
    print("R^2-Score is: {}".format(model.score(X_train_scaled, init['y_train'])))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_train'], pred_train))))
    print("MAE: {}".format(mean_absolute_error(init['y_train'], pred_train)))
    print("")
    print("w/ cross-validation")
    print("R2-Score is: {}".format(model.score(X_test_scaled, init['y_test'])))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(init['y_test'], pred))))
    print("MAE: {}".format(mean_absolute_error(init['y_test'], pred)))