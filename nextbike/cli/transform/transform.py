import click

from tqdm import tqdm
from time import sleep

from nextbike.preprocessing.Preprocessing import get_trip_data
from nextbike.io.output import __save_trip_data
from nextbike.constants import HEAD


HELP_WEATHER = '''
You can specify if the transformed file should contain weather information or not.
Weather-information is included (default).
You can set '--nw' if you don't want information about the weather included in the transformation 
'''

HELP_NAME = '''
You can specify a unique name for the transformed file e.g. 'transformed_data.csv'
If you don't specify a unique name, we name this file 'dortmund_transformation.csv'
'''


@click.command()
@click.argument('path', type=click.Path('rb'))
@click.option('--w/--nw', default=True, help=HELP_WEATHER)
@click.option('--name', '--n', default='dortmund_transformed.csv', help=HELP_NAME)
def transform(path, name, w):
    """
    This method transforms the data into the right format.
    You need to specify the path of the file, which needed to be transformed.
    You need to pass the PATH in this format 'foo/bar/test.csv'

    You can drop your file in the following location: 'data/internal/'
    You can find the new file under this location 'data/output/

    """

    path = HEAD + path
    click.echo('Dataframe loaded from ' + path)
    if w:
        df = get_trip_data(path, withWeather=True, head=True)
        click.echo('Information about the weather is included in the transformation')
    else:
        df = get_trip_data(path)
    click.echo('Dataframe has been transformed')
    __save_trip_data(df, name)
    click.echo('Dataframe saved to data/output/' + name)
