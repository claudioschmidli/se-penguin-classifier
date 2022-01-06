"""Flask app for penguin classification."""
import os
import pickle
from logging.config import dictConfig

import numpy as np
from flask import Flask, render_template, request


def initialize_logging():
    """Initialize the logger for the app."""
    LOGS_PATH = os.path.join(PARENT_DIR, "logs")
    if not os.path.exists(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    INFO_LOG_PATH = os.path.join(LOGS_PATH, "app_info.log")
    ERROR_LOG_PATH = os.path.join(LOGS_PATH, "app_error.log")

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


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "../../model/data/model.pkl")
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
initialize_logging()
app = Flask(__name__)
model = pickle.load(open(MODEL_PATH, "rb"))


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
    float_features = [x for x in request.form.values()]
    app.logger.info(
        f"User input for prediction: Culmen length = {float_features[0]}, Culmen depth = {float_features[1]}"
    )
    # @Tipi ich verstehe nicht wie man das anderst Formatieren kann mit dem Logger.
    try:
        float_features = [float(x) for x in float_features]
    except ValueError:
        app.logger.error(
            "Input values could not be converted to floats. Prediction aborded.",
            exc_info=True,
        )
        output = """
            <div id=negative_result>
                <p>Error! Please enter digits values as inputs!</p>
            """
        return render_template("inputFeatures.html", prediction_text=output)

    # logger.info(f"Features: Culmen length: {float_features[0]} mm, Culmen depth: {float_features[1]} mm")

    features = [np.array(float_features)]

    prediction = model.predict(features)

    if prediction[0]:
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
    return render_template("inputFeatures.html", prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)
