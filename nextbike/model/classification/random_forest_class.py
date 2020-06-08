from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

import numpy as np
import time

algorithm, precision, recall, f1score, support, exetime, desc = [], [], [], [], [], [], []
dic = {
    'algorithm': algorithm,
    'precision': precision,
    'recall': recall,
    'f1score': f1score,
    "support": support,
    "exetime": exetime,
    "desc": desc
}


def __init__(df):
    y = df['tripLabel']
    X = df[['weekend', 'hour', 'distanceToUniversity', 'month', 'area_start']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X': X,
        'y' :y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled

    }


def train(init):
    mod = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_depth=10,
                                 max_features='auto', bootstrap=False)
    mod.fit(init['X_train'], init['y_train'])
    y_pred = mod.predict(init['X_test'])

    rfc_cv_score = cross_val_score(mod, init['X'], init['y'], cv=10)
    __get_result(rfc_cv_score, init['y_test'], y_pred)

    print('10 fold cross validation' + np.mean(cross_val_score(mod, init['X_train'], init['y_train'], cv=10)))


def explore(typ, init, df):

    if typ not in ['away', 'towards']:
        raise Exception('typ has to be either away or towards')
    if typ == 'away':
        y_away = df['awayFromUniversity']
        X_train, X_test, y_away_train, y_away_test = train_test_split(init['X'], y_away, test_size=0.3)

        start = time.time()
        mod = RandomForestClassifier()
        mod.fit(X_train, y_away_train)
        y_pred = mod.predict(X_test)
        end = time.time()
        rfc_cv_score = cross_val_score(mod, init['X'], y_away, cv=10)

        __get_result(rfc_cv_score=rfc_cv_score, y_test=y_away_test, y_pred=y_pred)

        algorithm.append("Random Forrest")
        exetime.append((end - start))
        desc.append("Predicts awayFromUniversity (complement)")
    else:
        y_towards = df['towardsUniversity']
        X_train, X_test, y_towards_train, y_towards_test = train_test_split(init['X'], y_towards, test_size=0.3)

        start = time.time()
        mod = RandomForestClassifier()
        mod.fit(X_train, y_towards_train)
        y_pred = mod.predict(X_test)
        end = time.time()

        rfc_cv_score = cross_val_score(mod, init['X'], y_towards, cv=10)

        __get_result(rfc_cv_score=rfc_cv_score, y_test=y_towards_test, y_pred=y_pred)

        algorithm.append("Random Forrest")
        exetime.append((end - start))
        desc.append("Predicts towardsUniversity (complement)")


def predict_trip_label(init, mod=RandomForestClassifier()):
    start = time.time()
    mod = mod
    mod.fit(init['X_train'], init['y_train'])
    y_pred = mod.predict(init['X_test'])
    end = time.time()

    rfc_cv_score = cross_val_score(mod, init['X'], init['y'], cv=10)

    __get_result(rfc_cv_score=rfc_cv_score, y_test=init['y_test'], y_pred=y_pred)
    algorithm.append("Random Forrest")
    exetime.append((end-start))
    if mod.n_estimators == 1000:
        desc.append("Optimized hyperparameters of model in index 0")
    elif mod.n_estimators == 400:
        desc.append("Optimized hyperparameters of model in index 2")
    elif mod.n_estimators == 200:
        desc.append("Optimized hyperparameters of model in index 4")
    else:
        desc.append("Predicts tripLabel")


def optimize_hyper_parameters_random_forest(init):
    X_train, X_test, y_train, y_test = train_test_split(init['X'], init['y'], test_size=0.3)

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


def __get_result(rfc_cv_score, y_test, y_pred):
    pre, re, f_score, sup = score(y_test, y_pred, average='weighted')

    precision.append(pre)
    recall.append(re)
    f1score.append(f_score)
    support.append(sup)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print('\n')

    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())



