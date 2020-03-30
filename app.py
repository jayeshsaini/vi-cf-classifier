import json
import os
from flask import Flask, jsonify, request
from classify import predict

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

@app.route('/')
def index():
    return "Visual Inspection Classifier"

@app.route('/classify', methods = ['GET', 'POST'])
def classify():
  data = json.loads(request.get_data())
  #print(data)
  if data == None:
    return 'Got None'
  else:
    image = eval(data["image"])
    #print(image)
    prediction = predict(image)
    result = prediction.split()

  return jsonify({'Assembly':result[0], 'Orientation':result[1], 'Face':result[2]})

if __name__ == '__main__':
  app.run(port=port, host='0.0.0.0')
