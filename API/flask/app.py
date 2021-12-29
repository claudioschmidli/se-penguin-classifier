"""Flask app for pengiin classification."""
import logging
import os
import pickle

import numpy as np
from flask import Flask, render_template, request

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "../../model/data/model.pkl")
app = Flask(__name__)
model = pickle.load(open(MODEL_PATH, "rb"))

# setup logging
logging.basicConfig(filename="error.log", level=logging.ERROR)


@app.route("/")
def home():
    """Render HTML string for landing page of the app.

    Returns:
        str: String with HTML code containing the landing page for the app.
    """
    return render_template("inputFeatures.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Predict pengiun species and return reuslt as html.

    Returns:
        str: Result represented as HTML string that can be displayed in a browser.
    """
    try:
        float_features = [float(x) for x in request.form.values()]
    except ValueError:
        app.logger.error("A error happend", exc_info=True)
        error_Message = "bitte Masse mit Gleitkommazahlen definieren"
        return render_template("inputFeatures.html", prediction_text=error_Message)

    app.logger.info("features", float_features, exc_info=True)

    # print("features", float_features)
    features = [np.array(float_features)]

    # print("features1", features)
    prediction = model.predict(features)

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
    if not prediction[0]:
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
