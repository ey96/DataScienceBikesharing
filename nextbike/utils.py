import os



def get_data_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "../data")):
        return os.path.join(os.getcwd(), "../data")
    else:
        raise FileNotFoundError


def cast_address_to_coord(street):
    try:
        from geopy.geocoders import Nominatim
    except ImportError as e:
        print(e)

    geo_locator = Nominatim(user_agent="http")
    loc = geo_locator.geocode(street)
    return loc

