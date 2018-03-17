#!flask/bin/python
import os.path
import pickle
import numpy
import json
import imageio
from flask import Flask, Response, jsonify, make_response, request
from face_utils import encode_faces, load_model, recong_face_c, generate_response, search_face_by_distance
from make_classifier import make_classifier

app = Flask(__name__)

global i, j
i = 0
j = 0

@app.route('/get', methods=['GET'])
def get():
    return str(i) + "," + str(j)

@app.route('/update', methods=['GET'])
def update():
    global i, j
    i += 1
    j -= 1
    return "ok"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
