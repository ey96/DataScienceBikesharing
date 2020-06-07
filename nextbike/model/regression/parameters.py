from sklearn.linear_model import Ridge, Lasso

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