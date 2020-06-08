import click
from halo import Halo
from datetime import datetime
import os
from pathlib import Path
from nextbike.constants import CONSTANTS


from nextbike.constants import HEAD
from nextbike.preprocessing import get_trip_data
from nextbike.model.clustering import cluster

from nextbike.model.ensemble import random_forest
from nextbike.model.classification import random_forest_class
from nextbike.io.input import read_file
# define a spinner
spinner = Halo(text='Loading', spinner='dots')


@click.command()
@click.argument('path', type=click.Path('rb'))
@click.argument('model_type', type=click.Choice(['duration', 'destination', 'both'], case_sensitive=False))
def predict(path, model_type):

    spinner.start()
    spinner.succeed('[' + datetime.now().strftime('%H:%M:%S') + ']' + ' 1/2 data was load successfully from ' + path)

    path = HEAD + path
    d = Path(__file__).resolve().parents[3]
    path_trip = os.path.join(d, CONSTANTS.PATH_OUTPUT.value + "dortmund_trips.csv")

    df_test = get_trip_data(path, with_weather=True, head=True)
    df_trips = read_file(path_trip)

    # added trip_label as an additional attribute
    df_trips = cluster.__get_X_scaled(df_trips)
    df_test = cluster.__get_X_scaled(df_test)

    if model_type == 'duration':
        spinner.info(
            '[' + datetime.now().strftime('%H:%M:%S') + ']' + ' prediction for duration started')
        random_forest.predict(df_trips, df_test)
        spinner.succeed(
            '[' + datetime.now().strftime('%H:%M:%S') + ']' + ' 2/2 prediction was successful')

    if model_type == 'destination':
        spinner.info(
            '[' + datetime.now().strftime('%H:%M:%S') + ']' + ' prediction for destination started')
        random_forest_class.predict(df_trips=df_trips, df_test=df_test)
        spinner.succeed(
            '[' + datetime.now().strftime('%H:%M:%S') + ']' + ' 2/2 prediction was successful')
    # both
    elif model_type == 'both':
        pass

    else:
        click.echo('done')

