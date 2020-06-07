from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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
