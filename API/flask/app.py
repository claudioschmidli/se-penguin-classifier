# Import modules
import numpy as np
from flask import Flask, request, render_template
import pickle
import logging
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(currentdir, "../../model/model.pkl")
app = Flask(__name__)
model = pickle.load(open(model_path, "rb"))

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
