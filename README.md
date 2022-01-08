# App for classifying penguins
###SE Project by   Claudio Schmidli and Andrea Siedmann


## To do's:
- [x] Create model
- [x] Build API
- [x] Create git repository and share it on Gitlab
- [x] Add pre-commit hooks
- [x] CI pipline
- [ ] Add unit tests for all more functions => Claudio
- [x] Implement logging => Andrea
- [x] Implement TOX => Andrea
- [ ] Full documentation in readme.md => Claudio
- [x] Code formating (doc strings, type hinting, object oriented programming) => Andrea
- [ ] Deploy model
- [ ] Add delight to the experience when all tasks are complete :tada:


### Penguin Species Predictor
This repository contains the code and data for predicting the species membership of a penguin based on its specific characteristics, including testing, logging and an interface-based web application.

## Description
The model used can predict whether or not a penguin belongs to a particular species based on its specific characteristics culmen length and depth. The default is the test for the species Adelie. The classification method of binary logistic regression is used for the prediction.

A web application was programmed as the user interface.

The application is programmed in Python. The web application is developed with Flask.
The data source is a penguin dataset published on Kaggle with information on species affiliation (species: Chinstrap, Adélie or Gentoo) and certain body masses (culmen length and depth, flipper length, body mass, ...). You can find a detailed description of the penguins dataset [here] (https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris)

## Installation
The project can be downloaded [here](https://gitlab.com/claudio.schmidli/das-software-engineering-projektarbeit). In `requirements.txt` you can find, what python packages are required to run the project.

## Usage


## Authors
Claudio Schmidli, Andrea Siedmann

## Procedure
Search for a matching data set:
    Criteria: Size, clarity, free availability, good description: decision for the penguin dataset from Kaggle
Search for a suitable model:
    Criteria: Classification model, not too complex, not too much computing power, descriptive results: binary linear regression
Transfer of the model from Jupyter notebook
Building a virtual environment
Creating a Gitlab repository
Creating the folder structure
Creating the requirements.txt
Decomposition into object-oriented functions and classes Modularisation
Saving the model
Unit tests added
Pre-commit with black, autoflake, flake8 created
Extension of functions with doc strings
Gitlab pipeline created
Created interface-based web application with Flask
Extension with logging and exception handling
Tox implemented
Various code cleanings, test and logging extensions implemented
ReadMe file completed
