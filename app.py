#!flask/bin/python
import os.path
import os
import cv2
import pickle
import numpy as np
import json
import imageio
import facenet
from flask import Flask, Response, jsonify, make_response, request
from face_utils import encode_faces, load_model, recong_face_c, generate_response, search_face_by_distance, crossCheckDict
from make_classifier import make_classifier

app = Flask(__name__)

class JSONNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONNumpyEncoder, self).default(obj)

app.json_encoder = JSONNumpyEncoder

@app.route('/detectFacesC', methods=['POST'])
def detect_face_c():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    posbs, bbs, recg_ids = recong_face_c(model, sess, graph, pnet, rnet, onet, img)

    return make_response(jsonify(generate_response(posbs, bbs, recg_ids), 201))

@app.route('/detectFacesD', methods=['POST'])
def detect_face_d():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    posbs = []
    boxes = []
    ids = []

    embs1, embs2, ist = crossCheckDict(embeddings)
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embs1, embs2, np.asarray(ist))
    print("tpr, fpr, accuracy: ", accuracy)

    for emb, box, _ in embeddings_boxes:
        id, pos = search_face_by_distance(embeddings, emb)
        posbs.append(pos)
        boxes.append(box)
        ids.append(id)
    return make_response(jsonify(generate_response(posbs, boxes, ids), 201))

def save_img(id, img):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    folder = os.path.join(img_dir, id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # img_decode = cv2.imdecode(img, cv2.IMREAD_COLOR)
    file_num = len(os.listdir(folder))
    file_name = str(file_num + 1) + '.jpg'
    # cv2.imwrite(os.path.join(folder, file_name), img_decode)
    imageio.imwrite(os.path.join(folder, file_name), img)

@app.route('/registerFace/<string:id>', methods=['POST'])
def register_face(id):
    img_file = request.files['file']
    img = imageio.imread(img_file)
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        tra_img = embeddings_boxes[0][2]
        save_img(id, tra_img)
        if id not in embeddings:
            embeddings[id] = []
        embeddings[id].append(embeddings_boxes[0][0])
        with open(embedding_dat_path, 'wb') as outfile:
            pickle.dump(embeddings, outfile)
        return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/registerFaces/<string:id>', methods=['POST'])
def register_faces(id):
    img_file = request.files['file']
    img = imageio.imread(img_file)
    embeddings_boxes = encode_faces(graph, sess, pnet, rnet, onet, img)
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        tra_img = embeddings_boxes[0][2]
        save_img(id, tra_img)
        if id not in embeddings:
            embeddings[id] = []
        embeddings[id].append(embeddings_boxes[0][0])
        return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/registerFacesDone', methods=['POST'])
def register_faces_done():
    with open(embedding_dat_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
    return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/classifyFace', methods=['POST'])
def classify_face():
    global model
    model = make_classifier(sess, graph, embeddings, classifier_filename)
    return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/evolve', methods=['POST'])
def evolve():
    global model
    model = make_classifier(sess, graph, embeddings, classifier_filename)
    return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/getIds', methods=['GET'])
def get_ids():
    return json.dumps(list(model.classes_))

@app.route('/removeFace', methods=['DELETE'])
def remove_faces():
    global embeddings
    embeddings = {}
    os.remove(embedding_dat_path)
    return make_response(jsonify({'ok': 'ok'}), 201)

@app.route('/removeFace/<string:id>', methods=['DELETE'])
def remove_face(id):
    global embeddings
    if id in embeddings:
        del embeddings[id]
        with open(embedding_dat_path, 'wb') as outfile:
            pickle.dump(embeddings, outfile)
        return make_response(jsonify({'ok': 'ok'}), 201)
    return make_response(jsonify({'error': 'invalid id'}), 403)

global model, sess, graph, embeddings
embedding_dat_path = './embedding.dat'
embeddings = {}
# model_path = '../models/20170511-185253'
model_path = '../models/vggface2-cl'
img_dir = './images'
classifier_filename = './my_classifier.pkl'
if os.path.exists(embedding_dat_path):
    with open(embedding_dat_path, 'rb') as infile:
        embeddings = pickle.load(infile)
model, sess, graph, pnet, rnet, onet = load_model(model_path, classifier_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, use_reloader=False)
