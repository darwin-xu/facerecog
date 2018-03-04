#!flask/bin/python
import os.path
import pickle

from flask import Flask, Response, jsonify, make_response, request
from face_utils import encode_faces, load_model, recong_face
from make_classifier import make_classifier

app = Flask(__name__)

@app.route('/detectFacesC', methods=['POST'])
def detect_face_c():
    img = request.files['file']
    # p, bbox, id = recong_face(model, sess, class_names, pnet, rnet, onet, img)
    return ""

@app.route('/detectFacesD', methods=['POST'])
def detect_face_d():
    return ""

@app.route('/registerFace/<string:id>', methods=['POST'])
def register_face(id):
    img = request.files['file']
    embeddings_boxes = encode_faces(img)
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        embeddings[id].append(embeddings_boxes[0][0])
        # Saving embeddings
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
model, sess, graph, class_names, pnet, rnet, onet = load_model('models/facenet/20180220-152437', classifier_filename)

if __name__ == '__main__':
    app.run(debug=True)
