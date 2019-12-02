# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    b = pd.DataFrame.from_dict(data)
    
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(b)

    # Take the first value of prediction
    output = {'results': str(prediction)}

    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
