# Penguin Species Predictor
SE Project by Claudio Schmidli and Andrea Siedmann

This repository contains the code and data for predicting the species membership of a penguin based on its specific characteristics, including testing, logging and an interface-based web application.

## Description
The model used can predict whether or not a penguin belongs to a particular species based on its specific characteristics culmen length and depth. The default is the test for the species Adelie. The classification method of binary logistic regression is used for the prediction.

A web application was programmed as the user interface.

The application is programmed in Python. The web application is developed with Flask.

The data source is a penguin dataset published on Kaggle with information on species affiliation (species: Chinstrap, Adélie or Gentoo) and certain body masses (culmen length and depth, flipper length, body mass, ...). You can find a detailed description of the penguins dataset [here](https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris)

## Installation
The project can be downloaded [here](https://gitlab.com/claudio.schmidli/das-software-engineering-projektarbeit). In `requirements.txt` you can find, what python packages are required to run the project.

## How to use this project
In the file "classifier.py" the linear regression model is trained with the data from the penguin dataset and then saved as a PKL file. In addition, evaluation data on the quality of the model is generated. The penguin species to be examined can also be adjusted here.

The web application is connected to the stored model in "app.py" via Flask. Flask also handles deployment and hosting. When executing the file, the Flask Server is started. If the Flask server is running the Flask application will route to the default URL path.

The folder 'tests' contains the unit and integration tests. For Testing the Pytest framework was used. In these tests the correctness of the applications functionality and the model metrics are tested.

The GitLab pipeline for this project is configured in "gitlab-ci.yml". More information on pipeline architectures can be found [here](https://docs.gitlab.com/ee/ci/pipelines/pipeline_architectures.html).

Pre-commit hooks are used to ensure high commit quality. They are configured in ".pre-commit-config.yml". Further details on pre-commit can be found [here](https://pre-commit.com/).


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
- Saving the model
- Unit tests added
- Pre-commit with black, autoflake, flake8 created
- Extension of functions with doc strings
- Gitlab pipeline created
- Created interface-based web application with Flask
- Extension with logging and exception handling
- Tox implemented
- Various code cleanings, test and logging extensions implemented
- monitoring with app-usage with info-log
- furhter unit test and integration tests implemented
- ReadMe file completed

## Challenges
- resolving merge-conflicts
- logging Flask link not in logging-File
