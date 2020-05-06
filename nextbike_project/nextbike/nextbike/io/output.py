import os
import pickle


def save_model(model: object) -> object:
    pickle.dump(model, open(os.path.join(os.getcwd(), 'data/output/model.pkl'), 'wb'))
