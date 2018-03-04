#!flask/bin/python
import os.path
import pickle
import json

from flask import Flask, Response, jsonify, make_response, request
from face_utils import encode_faces, load_model, recong_face_c
from make_classifier import make_classifier

app = Flask(__name__)

def generate_response(posbs, bbs, recg_ids):
    response = {}
    response['faces'] = []
    for i in range(len(posbs)):
        face = {}
        face['possibility'] = posbs[i]
        face['x1'] = bbs[i][0]
        face['y1'] = bbs[i][1]
        face['x2'] = bbs[i][2]
        face['y2'] = bbs[i][3]
        face['id'] = recg_ids[i]
        response['faces'].append(face)
        
    return response

@app.route('/detectFacesC', methods=['POST'])
def detect_face_c():
    img = request.files['file']
    posbs, bbs, recg_ids = recong_face_c(model, sess, graph, ids, pnet, rnet, onet, img)

    return make_response(jsonify(generate_response(posbs, bbs, recg_ids), 201))

@app.route('/detectFacesD', methods=['POST'])
def detect_face_d():
    return ''

@app.route('/registerFace/<string:id>', methods=['POST'])
def register_face(id):
    img = request.files['file']
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
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
classifier_filename = '../models/my_classifier.pkl'
if os.path.exists(embedding_dat_path):
    with open(embedding_dat_path, 'rb') as infile:
        embeddings = pickle.load(infile)
model, sess, graph, ids, pnet, rnet, onet = load_model('models/facenet/20180220-152437', classifier_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
