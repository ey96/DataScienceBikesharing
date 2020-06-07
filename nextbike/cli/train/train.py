import click
from nextbike.constants import HEAD
from nextbike.model.ensemble import random_forest
from nextbike.model.clustering import cluster
from nextbike.preprocessing import get_trip_data

HELP_MODEL = """

DESCRIPTION HERE 

IF NOT.. THE BEST MODEL WILL BE CHOOSEN 

"""


@click.command()
@click.argument('path', type=click.Path('rb'))
@click.option('model-type', type=click.Choice(['duration', 'destination', 'both'], case_sensitive=False),
              help=HELP_MODEL)
def train(path, model):
    """
    DESCRIPTION HERE
    """
    if model == 'duration':
        file = HEAD + path
        df = get_trip_data(file, with_weather=True, head=True)
        click.echo('Dataframe is loaded from ' + file)
        df = cluster.__get_X_scaled(df)
        init = random_forest.__init__(df)
        random_forest.train(init)
    if model == 'destination':
        pass
    else:
        pass

