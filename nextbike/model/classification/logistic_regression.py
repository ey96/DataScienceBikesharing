from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score, classification_report
from sklearn.decomposition import PCA

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
    # use only start-information to classify the trip-class
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
        'X_test_scaled' : X_test_scaled

    }


def train(typ, df, init):
    pca_model = PCA(n_components=4).fit(init['X'])
    X_pca = pca_model.transform(init['X'])

    y_away = df['awayFromUniversity']
    y_towards = df['towardsUniversity']

    if typ not in ['away', 'towards']:
        raise Exception('typ has to be either away or towards')
    if typ == 'away':
        X_train, X_test, y_away_train, y_away_test = train_test_split(X_pca, y_away, test_size=0.3)

        start = time.time()
        mod = linear_model.LogisticRegression(C=0.12648552168552957, class_weight=None, dual=False,
                                              fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                                              random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                                              warm_start=False)

        mod.fit(init['X_train_scaled'], y_away_train)
        y_pred = mod.predict(init['X_test_scaled'])
        end = time.time()
        __get_result(y_away_test, y_pred)
        algorithm.append('Binary Logistic Regression')
        exetime.append((end - start))
        desc.append("Optimized hyper-parameters of model in index 0")

    else:
        X_train, X_test, y_towards_train, y_towards_test = train_test_split(init['X'], y_towards, test_size=0.3)
        start = time.time()
        mod = linear_model.LogisticRegression(C=0.8286427728546842, class_weight=None, dual=False,
                                              fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                                              random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                                              warm_start=False)
        mod.fit(init['X_train_scaled'], y_towards_train)
        y_pred = mod.predict(init['X_test_scaled'])
        end = time.time()

        __get_result(y_towards_test, y_pred)
        algorithm.append('Binary Logistic Regression')
        exetime.append((end - start))
        desc.append("Optimized hyperparameters of model in index 1")


def explore(model, df, init, typ=None, mod=linear_model.LogisticRegression()):
    if model not in ['Binary', 'Multilinear']:
        raise Exception('model has to be either Binary or Multilinear')

    y_away = df['awayFromUniversity']
    y_towards = df['towardsUniversity']

    if model == 'Binary':
        if typ not in ['away', 'towards']:
            raise Exception('typ has to be either away or towards')
        if typ == 'away':

            X_train, X_test, y_away_train, y_away_test = train_test_split(init['X'], y_away, test_size=0.3)
            start = time.time()
            mod = mod
            mod.fit(init['X_train_scaled'], y_away_train)
            y_pred = mod.predict(init['X_test_scaled'])
            end = time.time()

            __get_result(y_away_test, y_pred)
            algorithm.append("Binary Logistic Regression")
            exetime.append((end - start))
            desc.append("Predicts awayFromUniversity (is complementary to 2nd model)")

        # typ == towards
        else:

            X_train, X_test, y_towards_train, y_towards_test = train_test_split(init['X'], y_towards, test_size=0.3)
            start = time.time()
            mod = mod
            mod.fit(init['X_train_scaled'], y_towards_train)
            y_pred = mod.predict(init['X_test_scaled'])
            end = time.time()

            __get_result(y_towards_test, y_pred)
            algorithm.append("Binary Logistic Regression")
            exetime.append((end - start))
            desc.append("Predicts awayFromUniversity (is complementary to 1st model)")
    else:
        start = time.time()
        mod = linear_model.LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
        mod.fit(init['X_train_scaled'], init['y_train'])
        y_pred = mod.predict(init['X_test_scaled'])
        end = time.time()

        __get_result(init['y_test'], y_pred)
        algorithm.append("Multinomial Logistic Regression")
        exetime.append((end - start))
        desc.append("Predicts tripLabel")


# How to optimize hyper-parameters of a Logistic Regression model using Grid Search in Python
def optimize_hyper_parameters(X, y):
    # Create an scaler object
    sc = StandardScaler()

    # Create a pca object
    pca = decomposition.PCA()

    # Create a logistic regression object with an L2 penalty
    logistic = linear_model.LogisticRegression()

    # Create a pipeline of three steps. First, standardize the data.
    # Second, transform the data with PCA.
    # Third, train a logistic regression on the data.
    pipe = Pipeline(steps=[('sc', sc),
                           ('pca', pca),
                           ('logistic', logistic)])

    # Create Parameter Space
    # Create a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
    n_components = list(range(1, X.shape[1] + 1, 1))
    # Create a list of values of the regularization parameter
    C = np.logspace(-4, 4, 50)
    # Create a list of options for the regularization penalty
    penalty = ['l1', 'l2']
    # Create a dictionary of all the parameter options
    # Note has you can access the parameters of steps of a pipeline by using '__’
    parameters = dict(pca__n_components=n_components,
                      logistic__C=C,
                      logistic__penalty=penalty)

    # Conduct Parameter Optmization With Pipeline
    # Create a grid search object
    clf = GridSearchCV(pipe, parameters)

    # Fit the grid search
    clf.fit(X, y)
    # View The Best Parameters
    print('Best Penalty:', clf.best_estimator_.get_params()['logistic__penalty'])
    print('Best C:', clf.best_estimator_.get_params()['logistic__C'])
    print('Best Number Of Components:', clf.best_estimator_.get_params()['pca__n_components'])
    print()
    print(clf.best_estimator_.get_params()['logistic'])

    # Use Cross Validation To Evaluate Model
    CV_Result = cross_val_score(clf, X, y, cv=4, n_jobs=-1)
    print()
    print(CV_Result)
    print()
    print(CV_Result.mean())
    print()
    print(CV_Result.std())


def __get_result(y_test, y_pred):
    pre, re, f_score, sup = score(y_test, y_pred, average='weighted')
    print(classification_report(y_test, y_pred))

    precision.append(pre)
    recall.append(re)
    f1score.append(f_score)
    support.append(sup)


