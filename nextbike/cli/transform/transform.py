import click
from halo import Halo
from datetime import datetime

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

# define the spinner
spinner = Halo(text='Loading', spinner='dots')


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

    # start the spinner
    spinner.start()

    # adjust the path, since CLI is rooted from the root (i.e., ss_20_pds) and not from a sub-dir
    path = HEAD + path
    spinner.succeed('['+datetime.now().strftime('%H:%M:%S')+']'+' '+'1/3 data was load successfully from ' + path)

    # since the transformation step can take a couple if minutes, we want to give some visual feedback to the user
    spinner.warn('transform-data...this step can take a couple of minutes')

    # if the flag '--w' is active, we include weather-information in the transformation step
    if w:
        # we introduced a new parameter 'head' to deal with the needed resource of the CLI
        df = get_trip_data(path, with_weather=True, head=True)
        spinner.info('Information about the weather is included in the transformation')
    else:
        df = get_trip_data(path)

    spinner.succeed('['+datetime.now().strftime('%H:%M:%S')+']'+' '+'2/3 data has been transformed')

    # calls a private method, which saves the transformed file and give it a name
    __save_trip_data(df, name)
    spinner.succeed('['+datetime.now().strftime('%H:%M:%S')+']'+' '+'3/3 data successfully saved to data/output/ ' + name)

    # stop the spinner, since we are done
    spinner.stop()
