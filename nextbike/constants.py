from enum import Enum


class CONSTANTS(Enum):
    # path
    PATH_RAW = '../data/internal/'
    PATH_EXTERNAL = '../data/external/'
    PATH_PROCESSED = '../data/processed/'

    # map
    CENTER_OF_DORTMUND = [51.511838, 7.456943]
    TILES = "cartodbpositron"
    ATTR = '© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a ' \
           'href="http://cartodb.com/attributions#basemaps">CartoDB</a> '
