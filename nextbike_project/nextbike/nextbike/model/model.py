from .. import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def train():
    lr = LogisticRegression()
    lr.fit(X=io.read_file()[["sepal_length", "sepal_width", "petal_length", "petal_width"]], y=io.read_file()[["species"]])
    io.save_model(lr)
    return lr


def predict(sepal_length, sepal_widt, petal_length, petal_width):
    input_df = pd.DataFrame([[sepal_length, sepal_widt, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    lr = train()
    pred = lr.predict(input_df)
    return "Your input is a " + pred[0]

