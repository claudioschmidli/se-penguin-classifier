#Import modules
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('../../model/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('inputFeatures.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    print('features', float_features)
    features = [np.array(float_features)]
    print('features1', features)
    prediction = model.predict(features)

    output = "Penguin is of the species Adelie"
    if prediction[0] == False:
        output = "Penguin is not of the species Adelie"

    return render_template('inputFeatures.html', prediction_text=output)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(list(data.values()))]])

    output = "Penguin is of the species Adelie"
    if prediction[0] == False:
        output = "Penguin is not of the species Adelie"
        
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)