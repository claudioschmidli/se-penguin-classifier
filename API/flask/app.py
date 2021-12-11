# Import modules
import numpy as np
from flask import Flask, request, render_template
import pickle
import logging


app = Flask(__name__)
model = pickle.load(open("../../model/model.pkl", "rb"))

# setup logging
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
        app.logger.info("A error happend", exc_info=True)
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
    app.run(debug=True)