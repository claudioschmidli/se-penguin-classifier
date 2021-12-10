import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'culmen_length':2, 'culmen_depth':9})

print(r.json())