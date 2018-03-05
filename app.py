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

class JSONNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JSONNumpyEncoder, self).default(obj)

app.json_encoder = JSONNumpyEncoder

@app.route('/detectFacesC', methods=['POST'])
def detect_face_c():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    posbs, bbs, recg_ids = recong_face_c(model, sess, graph, ids, pnet, rnet, onet, img)

    return make_response(jsonify(generate_response(posbs, bbs, recg_ids), 201))

@app.route('/detectFacesD', methods=['POST'])
def detect_face_d():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    posbs = []
    boxes = []
    ids = []
    for emb, box in embeddings_boxes:
        id, pos = search_face_by_distance(embeddings, emb)
        posbs.append(pos)
        boxes.append(box)
        ids.append(id)
    return make_response(jsonify(generate_response(posbs, boxes, ids), 201))

@app.route('/registerFace/<string:id>', methods=['POST'])
def register_face(id):
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        if id not in embeddings:
            embeddings[id] = []
        embeddings[id].append(embeddings_boxes[0][0])
        with open(embedding_dat_path, 'wb') as outfile:
            pickle.dump((embeddings), outfile)
        return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/classifyFace', methods=['POST'])
def classify_face():
    make_classifier(sess, graph, embeddings, classifier_filename)
    return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/removeFace/<string:id>', methods=['DELETE'])
def remove_face(id):
    return ""

embedding_dat_path = './embedding.dat'
embeddings = {}
model_path = '../models/20170511-185253'
classifier_filename = '../models/my_classifier.pkl'
if os.path.exists(embedding_dat_path):
    with open(embedding_dat_path, 'rb') as infile:
        embeddings = pickle.load(infile)
model, sess, graph, ids, pnet, rnet, onet = load_model(model_path, classifier_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, use_reloader=False)
