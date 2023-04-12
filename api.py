import os
import pickle
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import numpy as np

# use CORS to allow us to run our API and hit it with frontend app
app = Flask(__name__)
CORS(app)
api = Api(app)

# Require a parser to parse the POST request
parser = reqparse.RequestParser()
parser.add_argument("sepal_length")
parser.add_argument("sepal_width")
parser.add_argument("petal_length")
parser.add_argument("petal_width")

# Unpickel the model
if os.path.isfile("./logRegModel.pkl"):
  model = pickle.load(open("./logRegModel.pkl", "rb"))
else:
  raise FileNotFoundError

class Predict(Resource):
  def post(self):
    args = parser.parse_args()
    
    X = (np.array(
      [
        args["sepal_length"],
        args["sepal_width"],
        args["petal_length"],
        args["petal_width"]
      ]
    ).astype("float").reshape(1, -1))
    
    _y = model.predict(X)[0]
    
    return { "class": _y }

api.add_resource(Predict, "/predictIris")

if __name__ == "__main__":
  app.run(debug=True)