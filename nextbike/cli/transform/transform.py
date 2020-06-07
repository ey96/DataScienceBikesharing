import click

from tqdm import tqdm
from time import sleep

from nextbike.preprocessing.Preprocessing import get_trip_data
from nextbike.io.output import __save_trip_data
from nextbike.constants import HEAD


HELP_FILE = '''
You can specify a unique name for the transformed file e.g. 'transformed_data.csv'
If you don't specify a unique name, we name this file 'dortmund_transformation.csv'
'''


@click.command()
@click.argument('file', type=click.Path('rb'))
@click.option('--output', default='dortmund_transformed.csv', help=HELP_FILE)
def transform(path, output):
    """
    This method transforms the data into the right format.
    You need to specify the path of the file, which needed to be transformed.
    You need to pass the PATH in this format 'foo/bar/test.csv'

    You can drop your file in the following location: 'data/internal/'
    You can find the new file under this location 'data/output/

    """

    path = HEAD + path
    click.echo('Dataframe loaded from ' + path)
    df = get_trip_data(path)
    click.echo('Dataframe transformed')
    __save_trip_data(df, output)
    click.echo('Dataframe saved to data/output/' + output)
