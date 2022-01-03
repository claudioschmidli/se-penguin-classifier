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
logger = logging.getLogger(__name__)
logging.basicConfig(filename="error.log", level=logging.ERROR)


@app.route("/")
def home():
    return render_template("inputFeatures.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """

    try:
        float_features = [float(x) for x in request.form.values()]
    except ValueError:
        app.logger.error("A error happend", exc_info=True)
        error_Message = "bitte Masse mit Gleitkommazahlen definieren"
        return render_template("inputFeatures.html", prediction_text=error_Message)
    print("features", float_features)
    features = [np.array(float_features)]
    print("features1", features)
    prediction = model.predict(features)

    output = "Penguin is of the species Adelie"
    if not prediction[0]:
        output = "Penguin is not of the species Adelie"

    return render_template("inputFeatures.html", prediction_text=output)


if __name__ == "__main__":
    app.run()
