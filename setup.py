from setuptools import setup

setup(
    name='nextbike',
    version='0.0.1dev1',
    description="Programming Data Science",
    author="Students",
    author_email="blankspace@blankspace",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click', 'geopy', 'geopandas', 'numpy', 'folium', 'shapely', 'seaborn',
                      'matplotlib'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:main']
    }
)