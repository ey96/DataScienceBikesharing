![CI](https://github.com/t00m-dev/ss_20_pds/workflows/CI/badge.svg?branch=master)
# ss_20_pds
This repository contains all materials that were developed in the team for the lecture "Programming Data Science" at the University of Cologne.
## project structure 

    .
    ├── .github                   # contains internal setting e.g. PR/Issue -template, workflows    
    ├── data                      
    │   ├── external              # contains external data like weather, geojson etc.
    │   ├── internal              # contains the raw training and test-data
    │   ├── output                # contains the output from the CLI-commands
    │   └── processed             # contains proccessed csv-files 
    ├── doc                       
    │   ├── figures               # stores all the figures, which are gernerated through the analysis
    │   ├── Timeline_PDS2020.docx # our timeline
    │   └── report.pdf            # our group-report
    ├── nextbike                  # our python-project (i.e., src)
    ├── notebooks                                         
    │   ├── archiv                                      # contains all the 'plain'- notebooks 
    │   ├── Task 1 - Exploration and Description.ipynb      
    │   ├── Task 2 - Visualization.ipynb                
    │   └── Task 3 - Prediction.ipynb                   
    ├── README.md
    └── setup.py
    
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