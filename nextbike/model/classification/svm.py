from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, classification_report

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
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled

    }


def explore(model, typ, init, df):

    if model not in ['linear', 'rbf']:
        raise Exception('model has to be either linear or rbf')

    if typ not in ['away', 'towards']:
        raise Exception('typ has to be either away or towards')

    if typ == 'away':
        y_away = df['awayFromUniversity']
        X_train, X_test, y_away_train, y_away_test = train_test_split(init['X'], y_away, test_size=0.3)

        start = time.time()
        if model == 'linear':
            mod = SVC(kernel='linear')
            algorithm.append("SVM with linear kernel")
            desc.append("Predicts awayFromUniversity (is complementary to 2nd model)")
        else:
            mod = SVC(kernel='rbf')
            algorithm.append("SVM with rbf kernel")
            desc.append("Predicts awayFromUniversity (is complementary to 4th model)")

        mod.fit(init['X_train_scaled'], y_away_train)
        y_pred = mod.predict(init['X_test_scaled'])
        end = time.time()

        __get_result(y_away_test, y_pred)
        exetime.append((end - start))
    else:
        y_towards = df['towardsUniversity']
        X_train, X_test, y_towards_train, y_towards_test = train_test_split(init['X'], y_towards, test_size=0.3)

        start = time.time()
        if model == 'linear':
            mod = SVC(kernel='linear')
            algorithm.append("SVM with linear kernel")
            desc.append("Predicts towardsUniversity (is complementary to 1st model)")
        else:
            mod = SVC(kernel='rbf')
            algorithm.append("SVM with rbf kernel")
            desc.append("Predicts towardsUniversity (is complementary to 3rd model)")

        mod.fit(init['X_train_scaled'], y_towards_train)
        y_pred = mod.predict(init['X_test_scaled'])
        end = time.time()

        __get_result(y_towards_test, y_pred)
        algorithm.append("SVM with linear kernel")
        exetime.append((end - start))


def optimize_hyper_parameters(X, y, nfolds):
    # defining parameter range
    param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [0.1, 1, 10, 100, 1000],
                   'gamma': [1, 0.5, 0.3, 0.1, 0.01, 0.001, 0.0001],
                   'kernel': ['rbf']}]

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=nfolds)

    # fitting the model for grid search
    grid.fit(X, y)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)


def __get_result(y_test, y_pred):
    pre, re, f_score, sup = score(y_test, y_pred, average='weighted')
    print(classification_report(y_test, y_pred))

    precision.append(pre)
    recall.append(re)
    f1score.append(f_score)
    support.append(sup)


