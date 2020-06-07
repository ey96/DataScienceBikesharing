from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score

import numpy as np


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


