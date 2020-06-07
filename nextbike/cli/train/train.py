import click
from nextbike.constants import HEAD
from nextbike.model.regression.linear_regression import __init__, linear_regression
from nextbike.preprocessing import get_trip_data

HELP_MODEL = """

DESCRIPTION HERE 

IF NOT.. THE BEST MODEL WILL BE CHOOSEN 

"""


@click.command()
@click.argument('path', type=click.Path('rb'))
@click.option('model-type', type=click.Choice(['linear', 'polynomial', 'svr'], case_sensitive=False), help=HELP_MODEL)
def train(path, model):
    """
    DESCRIPTION HERE
    """

    ll = HEAD + path
    click.echo('Dataframe loaded from ' + ll)
    df = get_trip_data(ll, with_weather=True, head=True)
    init = __init__(df)
    linear_regression(init)
