import click
from halo import Halo
from datetime import datetime

from nextbike.constants import HEAD
from nextbike.model.ensemble import random_forest
from nextbike.model.clustering import cluster
from nextbike.preprocessing import get_trip_data
from nextbike.model.classification import random_forest_class


# define a spinner
spinner = Halo(text='Loading', spinner='dots')


@click.command()
@click.argument('path', type=click.Path('rb'))
@click.argument('model_type', type=click.Choice(['duration', 'destination', 'both'], case_sensitive=False))
def train(path, model_type):
    """
    Call this command if you want to train your model.
    After the 'training' the trained model is saved under 'data/output/'

    This command takes the path to the csv-file as an argument, as well as an argument 'model-type'
    The 'model-type' argument can be 'duration', 'destination' or 'both'.
    With this argument you can specify which kind of model you want to train.

    Due our limited computational power, we decide to introduce the flag,
    since the training of both-model take a couple of minutes

    ### FURTHER INFORMATION ###
    According to the project-description the 'duration-' model represents Task 3a
    and the 'destination-' model represents Task 3b

    """

    # start the spinner
    spinner.start()
    spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + '1/3 data was load successfully from' + path)

    file = HEAD + path

    df = get_trip_data(file, with_weather=True, head=True)

    # added trip_label as an additional attribute
    df = cluster.__get_X_scaled(df)

    if model_type == 'duration':
        spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + '2/3 training of duration-model started')
        spinner.warn('[ATTENTION] this step can take a couple of minutes')
        # call our training model
        init = random_forest.__init__(df)
        # train
        random_forest.train(init)
        spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + '3/3 training done. Model saved under '
                                                                          '/data/output')
    if model_type == 'destination':
        spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + '2/3 training of destination-model started')
        spinner.warn('[ATTENTION] this step can take a couple of minutes')
        # call out training model
        init = random_forest_class.__init__(df)
        # train
        random_forest_class.train(init)
        spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + '3/3 training done. Model saved under '
                                                                          '/data/output')
    # both
    else:
        print('hm')
        pass

    spinner.stop()
