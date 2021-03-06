from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

model_dic = {
    "Ridge": {
        "model": Ridge(),
        "parameters": {
            'alpha': [0.001, 0.01, 0.1, 2, 3, 4, 10, 20],
            'max_iter': [500, 1000, 2000, 3000],
            'random_state': [0, 1],
            "fit_intercept": [True, False],

        }
    },
    "Lasso": {
        "model": Lasso(),
        "parameters": {
            "alpha": [0.001, 0.01, 0.1, 1, 2, 3, 4, 10, 20],
            'max_iter': [500, 1000, 2000, 3000],
            "random_state": [0, 1],
            "fit_intercept": [True, False]
        }
    }
}

svr = SVR()
kernel = ["poly", "rbf", "linear", "sigmoid"]
C = [10, 20, 40, 80, 150, 200]
epsilon = [10, 20, 30, 40, 50]
degree = [1, 2, 3, 4]
gamma = ["auto", "scale"]
verbose = [True, False]
max_iter = [10, 20, 30, 40, 50, 100, 200, 500]

random_grid = {'kernel': kernel,
               'C': C,
               'epsilon': epsilon,
               'gamma': gamma,
               "degree": degree,
               "verbose": verbose,
               "max_iter": max_iter
}

ridge = Ridge()
alpha = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]
copy = [True, False]
fit_intercept = [True, False]
max_iter = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

random_grid_poly = {'alpha': alpha, 'copy_X': copy, 'fit_intercept': fit_intercept, 'max_iter': max_iter,
                    'solver': solver}

