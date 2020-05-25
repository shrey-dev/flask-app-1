# Main Application file

# Importing libraries
import pandas as pd
from flask import Flask, jsonify, request
import pickle
import json
import requests

# loading model
model = pickle.load(open('model.pkl', 'rb'))

# Creating app
app = Flask(__name__)

# Creating Routes
@app.route('/', methods=['POST'])

# Making prediction function
def predict():
    data = request.get_json(force=True)
    #converting data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    #predictions
    result = model.predict(data_df)
    #sending back to browser
    output = {'results : ': int(result[0])}
    #return data
    return jsonify(result=True)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

# sample data
url = 'http://127.0.0.1:5000'
data = {'Pclass': 3, 'Age': 2, 'SibSp': 1, 'Fare': 50}
data = json.dumps(data)

send_req = requests.post(url, data)
print(send_req)