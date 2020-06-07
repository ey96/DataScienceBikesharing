![CI](https://github.com/t00m-dev/ss_20_pds/workflows/CI/badge.svg?branch=master)
# ss_20_pds
This repository contains all materials that were developed in the team for the lecture "Programming Data Science" at the University of Cologne.

## Command Line Interface (CLI)

#### 1. How to install the project? 
CLI root of the project

```console
foo@bar:~ss_20_pds$ pip3 install .
``` 

```console
foo@bar:~ss_20_pds$ pip3 install -e .
```


#### Transformation 

##### 1. Help
```console
foo@bar:~ss_20_pds$ nextbike transform --help

  This method transforms the data into the right format. You need to specify
  the path of the file, which needed to be transformed. You need to pass the
  PATH in this format 'foo/bar/test.csv'

  You can drop your file in the following location: 'data/internal/' You can
  find the new file under this location 'data/output/

Options:
  --w / --nw        You can specify if the transformed file should contain weather information or not.
                    Weather-information is included (default).
                    You can set '--nw' if you don't want information about the weather included in the transformation
  --name, --n TEXT  You can specify a unique name for the transformed file
                    e.g. 'transformed_data.csv'
                    If you don't specify a unique name, we name this file 'dortmund_transformation.csv'
  --help            Show this message and exit.
```

##### 2. Example
```console
foo@bar:~ss_20_pds$ nextbike transform data/internal/dortmund.csv
```

```console
foo@bar:~ss_20_pds$ nextbike transform data/internal/dortmund.csv --nw --name no_weather_example_name.csv
```

#### Train

```console
foo@bar:~ss_20_pds$ nextbike train data/internal/dortmund.csv
```