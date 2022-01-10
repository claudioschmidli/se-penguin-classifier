"""Flask app for penguin classification."""
import configparser
import os
import pickle
from logging.config import dictConfig
from typing import Optional, Union

import numpy as np
import sklearn.linear_model
from flask import Flask, render_template, request


def initialize_logging():
    """Initialize the logger for the app."""
    config = configparser.ConfigParser()
    config.read(os.path.join(CURRENT_DIR, "config.ini"))
    LOGS_PATH = config["LOGGING"]["LOGS_PATH"]
    if not os.path.exists(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    INFO_LOG_PATH = config["LOGGING"]["INFO_LOG_FILE"]
    INFO_LOG_PATH = os.path.join(LOGS_PATH, INFO_LOG_PATH)
    ERROR_LOG_PATH = config["LOGGING"]["ERROR_LOG_FILE"]
    ERROR_LOG_PATH = os.path.join(LOGS_PATH, ERROR_LOG_PATH)

    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                }
            },
            "disable_existing_loggers": False,
            "handlers": {
                "file_info": {
                    "class": "logging.FileHandler",
                    "filename": INFO_LOG_PATH,
                    "level": "INFO",
                    "formatter": "default",
                },
                "file_error": {
                    "class": "logging.FileHandler",
                    "filename": ERROR_LOG_PATH,
                    "level": "ERROR",
                    "formatter": "default",
                },
                "console_info": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://flask.logging.wsgi_errors_stream",
                    "formatter": "default",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["file_info", "file_error", "console_info"],
            },
        }
    )


def get_model() -> sklearn.linear_model._logistic.LogisticRegression:
    """Load linear model for penguin classification from a pickle file.

    Returns:
        sklearn.linear_model._logistic.LogisticRegression: Linear model for penguin classification.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(CURRENT_DIR, "config.ini"))
    MODEL_PATH = config["CLASSIFICATION"]["MODEL_PATH"]
    MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_PATH)
    model = pickle.load(open(MODEL_PATH, "rb"))
    return model


def create_app() -> Flask:
    """Create flask app.

    Returns:
        Flask: Instance of a flask class
    """
    os.path.join(CURRENT_DIR, "config.ini")
    TEMPLATE_FOLDER = os.path.join(CURRENT_DIR, "templates")
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
    return app


def render_result(
    prediction: Union[np.ndarray, str],
    float_features: Optional[list] = [
        None,
    ],
) -> str:
    """Format prediction result to HTML string.

    Args:
        prediction (Union[np.ndarray, str]): Variable containing outcome of the prediction. (if error use ="error")
        float_features (Optional[list], optional): List containing user input features. Defaults to [None, ].

    Returns:
        str: Result of prediction formated to a HTML string.
    """
    print("")
    if type(prediction) == str and prediction == "error":
        app.logger.error(
            "Input values could not be converted to floats. Prediction aborded.",
            exc_info=True,
        )
        output = """
            <div id=negative_result>
                <p>Error! Please enter digits values as inputs!</p>
            """
        return output
    elif prediction[0]:
        app.logger.info("The peguin was predicted as Adelie.")
        output = f"""
        <div id=positive_result>
            <p>Penguin is of the species Adelie</p>
            <ul>
                <li>Culmen length: {float_features[0]} mm</li>
                <li>Culmen depth: {float_features[1]} mm</li>
            </ul>
            <img src="static/adelie.png" alt="Yes it is Adelie">
        </div>
        """
        return output
    else:
        app.logger.info("The peguin was predicted other than Adelie.")
        output = f"""
        <div id=negative_result>
            <p>Penguin is not of the species Adelie</p>
            <ul>
                <li>Culmen length: {float_features[0]} mm</li>
                <li>Culmen depth: {float_features[1]} mm</li>
            </ul>
            <img src="static/no_adelie.png" alt="No it is not Adelie">
        </div>
        """
        return output


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
initialize_logging()
app = create_app()


@app.route("/")
def home():
    """Render HTML string for landing page of the app.

    Returns:
        str: String with HTML code containing the landing page for the app.
    """
    app.logger.info("Load landing page.")
    return render_template("inputFeatures.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Predict pengiun species and return the result as html.

    Returns:
        str: Result represented as HTML string that can be displayed in a browser.
    """
    print(request.form.values())
    float_features = [x for x in request.form.values()]
    app.logger.info(
        f"User input for prediction: Culmen length = {float_features[0]}, Culmen depth = {float_features[1]}"
    )
    # @Tipi ich verstehe nicht wie man das anderst Formatieren kann mit dem Logger.
    try:
        float_features = [float(x) for x in float_features]
    except ValueError:
        prediction = "error"
        output = render_result(prediction)
        return render_template("inputFeatures.html", prediction_text=output)

    features = [np.array(float_features)]
    model = get_model()
    prediction = model.predict(features)
    output = render_result(prediction, float_features)
    output = render_template("inputFeatures.html", prediction_text=output)
    return output


if __name__ == "__main__":
    app.run(debug=True)
