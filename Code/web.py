from flask import Flask, request,jsonify
import numpy as np
from predict import *

import json
from flask_cors import cross_origin
app = Flask(__name__)

pokemon = Pokemon()

@app.route("/", methods=['GET'])
@cross_origin()
def predict():
    columns = ['latitude', 'longitude', 'appearedTimeOfDay', 'appearedDayOfWeek',
       'terrainType', 'closeToWater', 'continent', 'temperature',
       'weatherIcon', 'urban', 'rural', 'gymDistanceKm', 'pokestopDistanceKm']
    data = []
    for col in columns:
        print(request.args.get(col))
        data.append(request.args.get(col))
    data = convert(data)
    result = pokemon.get_ids(data).tolist()
    res = {'ids':result}
    return jsonify(res)

if __name__ == '__main__':
    app.run(threaded=False)



