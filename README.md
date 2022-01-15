# Penguin Species Predictor
SE Project by Claudio Schmidli and Andrea Siedmann

This repository contains the code and data for predicting the species membership of a penguin based on its specific characteristics, including testing, logging and an interface-based web application.

## Description
The model used can predict whether or not a penguin belongs to a particular species based on its specific characteristics culmen length and depth. The default is the test for the species Adelie. The classification method of binary logistic regression is used for the prediction.

A web application was programmed as the user interface. Please have a look at the online demo [here](https://penguin-classifier1-app.herokuapp.com/)(the first time loading the page takes several seconds in order to start up the app).

The application is programmed in Python. The web application is developed with Flask.

The data source is a penguin dataset published on Kaggle with information on species affiliation (species: Chinstrap, Adélie or Gentoo) and certain body masses (culmen length and depth, flipper length, body mass, ...). You can find a detailed description of the penguins dataset [here](https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris).

## Installation
The project can be downloaded [here](https://gitlab.com/claudio.schmidli/das-software-engineering-projektarbeit). In `requirements.txt` you can find, what python packages are required to run the project.

## How to use this project

#### Model
In the file `model\classifier.py` the linear regression model is trained with the data from the penguin dataset and then saved as a PKL file. In addition, evaluation data on the quality of the model is generated. The penguin species to be examined can also be adjusted here.

#### Web app
The web application is connected to the stored model in `\API\flask\app.py` via Flask. Flask also handles deployment and hosting. When executing the file, the Flask Server is started. If the Flask server is running the Flask application will route to the default URL path.

#### Testing
For testing you can start the application from the root directory using the following command:

 - `python \API\flask\app.py`

The folder 'tests' contains the unit and integration tests. The Pytest framework was used for testing the applications functionality and the model metrics. Run all the tests by using the following command:

- `pytest`

#### CI pipline
The GitLab pipeline for this project is configured in "gitlab-ci.yml". More information on pipeline architectures can be found [here](https://docs.gitlab.com/ee/ci/pipelines/pipeline_architectures.html). The configured CI pipline starts tox after pushing files to Gitlab which will perform the tests in a docker container.

#### Pre-commit hooks
Pre-commit hooks are used to ensure high commit quality. They are configured in ".pre-commit-config.yml". Further details can be found [here](https://pre-commit.com/). 

The following packages are set up:

- pre-commit-hooks
- black
- autoflake
- flake8
- isort
- pydocstyle

#### Logging
The app contains a logger which writes entries onto the command line as well as in local files in `/logs`. The different files contain information of the log levels `INFO` and `ERROR`.



## Authors
Claudio Schmidli, Andrea Siedmann

## Procedure
- Search for a matching data set: <br>
Criteria: Size, clarity, free availability, good description: decision for the penguin dataset from Kaggle
- Search for a suitable model: <br>
Criteria: Classification model, not too complex, not too much computing power, descriptive results: binary linear regression
- Transfer of the model from Jupyter notebook
- Building a virtual environment
- Creating a Gitlab repository
- Creating the folder structure
- Creating the requirements.txt
- Decomposition into object-oriented functions and classes, modularisation
- Saving the model into a pickle file
- Unit tests added
- Pre-commit with black, autoflake, flake8, isort, pydocstyle created
- Extension of functions with doc strings
- Gitlab pipeline created
- Created interface-based web application with Flask
- Extension with logging and exception handling
- Tox implemented
- Various code cleanings, test and logging extensions implemented
- monitoring with app-usage with info-log
- further unit test and integration tests implemented
- ReadMe file completed
- added branch for deployment to Heroku 
- Deployment to [Heroku](https://penguin-classifier1-app.herokuapp.com/).

## Challenges
- resolving merge-conflicts (example for merging two branches on 10 Jan, 2022, Commit 2b3f1207)
- logging Flask app URL and other log entries into local log files as well as onto the command line interface
- inplement type hints and add correct data types to hints
- write clean code meeting PEP8 criteria (pylint suggestions are often hard to implement)
- import own local modules from different directories
- write a functional test for flask by using a test client
