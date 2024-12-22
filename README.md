<h1 align="center">Penguin Species Predictor</h1>
<h3 align="center">Course Project @ FHNW</h3>
<h3 align="center">Software Engineering for Data Scientists</h3>
<h4 align="center">Claudio Schmidli & Andrea Siedmann</h4>

## Academic Context
This project was developed as part of the Software Engineering for Data Scientists course at FHNW (University of Applied Sciences and Arts Northwestern Switzerland). The course emphasized professional software engineering practices in the context of machine learning applications.

### Software Engineering Focus
- REST API development and implementation using Flask
- Type annotation and variable documentation for code clarity
- Git version control and collaborative development practices
- Automated code quality enforcement via pre-commit hooks
- Comprehensive testing strategies (unit and integration testing)
- Clean code principles (PEP8 compliance)
- Continuous Integration/Continuous Deployment (CI/CD)

## Project Overview
The project implements a machine learning model for penguin species prediction, emphasizing software engineering best practices. The model predicts whether a penguin belongs to the Adelie species based on its physical characteristics (culmen length and depth).

### Key Features
- REST API interface for model predictions
- Automated testing pipeline with pytest and tox
- Comprehensive logging system
- Code quality automation through pre-commit hooks
- GitLab CI/CD pipeline integration
- Web-based user interface

### Technical Stack
- **Backend**: Python, Flask
- **API**: REST architecture
- **Machine Learning**: Binary Logistic Regression
- **Testing**: 
  - pytest (unit and integration tests)
  - tox (test automation)
- **Code Quality**: 
  - Black (code formatting)
  - Flake8 (style guide enforcement)
  - isort (import sorting)
  - pydocstyle (documentation checking)
  - autoflake (dead code removal)
- **CI/CD**: GitLab CI
- **Deployment**: Heroku

_Note: This project was originally developed and hosted on GitLab as part of the course requirements._