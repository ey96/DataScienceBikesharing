import pathlib
from pathlib import Path
from enum import Enum


HEAD = 'ss_20_pds/'
__FILE__ = path = Path(pathlib.Path().absolute()).parent


class CONSTANTS(Enum):
    # path
    PATH_RAW = 'data/internal/'
    PATH_EXTERNAL = 'data/external/'
    PATH_PROCESSED = 'data/processed/'
    PATH_OUTPUT = 'data/output/'

    # map
    CENTER_OF_DORTMUND = [51.511838, 7.456943]
    TILES = "cartodbpositron"
    ATTR = '© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a ' \
           'href="http://cartodb.com/attributions#basemaps">CartoDB</a> '
