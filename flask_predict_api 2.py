
"""

import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('/Users/WhiyongL/Desktop/flask and docker/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict',methods=["GET"])
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: Return type of iris
            schema:
                type: number  
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: Return type of iris
            schema:
                type: array
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#http://localhost:5000/predict?s_length=5.7&s_width=5.6&p_length=4.3&p_width=7.8
#http://localhost:5000/predict?s_length=5.9&s_width=3&p_length=5.1&p_width=1.8
    
#http://localhost:5000/apidocs/
    
    
    
    
    
    
    
    
    
    
    
    
    
