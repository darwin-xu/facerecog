import json
import os
import os.path
import pickle
import shutil
import sys
import time

import imageio
import numpy as np
from flask import Flask, Response, jsonify, make_response, request

import face_utils
import facenet
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

global model, sess, graph, embeddings, img_dir

threshold = 0
embedding_dat_path = './embedding.dat'
embeddings = {}
model_path = os.path.abspath(sys.argv[1])
model_name = os.path.basename(model_path)
classifier_filename = './my_classifier.pkl'
model, sess, graph, pnet, rnet, onet = face_utils.load_model(
    model_path, classifier_filename)

img_dir = './images'
detect_img_dir = './detect_images'
emb_model_name = ''
if os.path.exists(embedding_dat_path):
    with open(embedding_dat_path, 'rb') as infile:
        try:
            embeddings = pickle.load(infile)
            emb_model_name = pickle.load(infile)
        except EOFError:
            pass
if model_name != emb_model_name:
    print(
        'Embedding is not compatible with the model, recalculate.', flush=True)
    embeddings = face_utils.freshEmbedding(graph, sess, img_dir)
    with open(embedding_dat_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
        pickle.dump(model_name, outfile)
else:
    print('Embedding is compatible with the model.', flush=True)


########################################################################################
def save_detect_img(id, img):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    folder = os.path.join(img_dir, id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_num = len(os.listdir(folder))
    file_name = str(file_num + 1) + '.jpg'
    imageio.imwrite(os.path.join(folder, file_name), img)


@app.route('/detectFacesC', methods=['POST'])
@face_utils.timed
def detect_face_c():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    posbs, bbs, recg_ids = face_utils.recong_face_c(model, sess, graph, pnet,
                                                    rnet, onet, img)

    for i in range(len(posbs)):
        img_cropped = img[bbs[i][1]:bbs[i][3], bbs[i][0]:bbs[i][2], :]
        save_img(recg_ids[i], img_cropped, detect_img_dir, posbs[i])
    sys.stdout.flush()
    return make_response(
        jsonify(face_utils.generate_response(posbs, bbs, recg_ids), 201))


@app.route('/detectFacesD', methods=['POST'])
@face_utils.timed
def detect_face_d():
    imgFile = request.files['file']
    img = imageio.imread(imgFile)
    embeddings_boxes = face_utils.encode_faces(graph, sess, pnet, rnet, onet,
                                               img)
    posbs = []
    boxes = []
    ids = []

    global threshold
    if threshold == 0:
        embs1, embs2, ist = face_utils.crossCheckDict(embeddings)
        thresholds = np.arange(0, 4, 0.01)
        _, _, accuracy, threshold = facenet.calculate_roc(
            thresholds, embs1, embs2, np.asarray(ist))
        print("accuracy: ", accuracy)
        print("partial thresholds:", threshold)
        sys.stdout.flush()

    for emb, box, _ in embeddings_boxes:
        id, pos = face_utils.search_face_by_distance(embeddings, emb,
                                                     threshold)
        posbs.append(pos)
        boxes.append(box)
        ids.append(id)
        img_cropped = img[box[1]:box[3], box[0]:box[2], :]
        save_img(id, img_cropped, detect_img_dir, pos)
    sys.stdout.flush()
    return make_response(
        jsonify(face_utils.generate_response(posbs, boxes, ids), 201))


def save_img(id, img, parent=img_dir, possibility=0):
    if not os.path.exists(parent):
        os.makedirs(parent)
    folder = os.path.join(parent, id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_num = len(os.listdir(folder))
    file_name = str(file_num + 1) + '_' + str(possibility) + '.jpg'
    imageio.imwrite(os.path.join(folder, file_name), img)


@app.route('/registerFace/<string:id>', methods=['POST'])
@face_utils.timed
def register_face(id):
    img_file = request.files['file']
    img = imageio.imread(img_file)
    embeddings_boxes = face_utils.encode_faces(graph, sess, pnet, rnet, onet,
                                               img)
    sys.stdout.flush()
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        global threshold
        threshold = 0
        tra_img = embeddings_boxes[0][2]
        save_img(id, tra_img)
        if id not in embeddings:
            embeddings[id] = []
        embeddings[id].append(embeddings_boxes[0][0])
        with open(embedding_dat_path, 'wb') as outfile:
            pickle.dump(embeddings, outfile)
            pickle.dump(model_name, outfile)
        return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/registerFaces/<string:id>', methods=['POST'])
@face_utils.timed
def register_faces(id):
    img_file = request.files['file']
    img = imageio.imread(img_file)
    embeddings_boxes = face_utils.encode_faces(graph, sess, pnet, rnet, onet,
                                               img)
    sys.stdout.flush()
    if len(embeddings_boxes) != 1:
        return make_response(jsonify({'error': 'invalid image'}), 403)
    else:
        global threshold
        threshold = 0
        tra_img = embeddings_boxes[0][2]
        save_img(id, tra_img)
        if id not in embeddings:
            embeddings[id] = []
        embeddings[id].append(embeddings_boxes[0][0])
        return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/registerFacesDone', methods=['POST'])
@face_utils.timed
def register_faces_done():
    with open(embedding_dat_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
        pickle.dump(model_name, outfile)
    return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/classifyFace', methods=['POST'])
@face_utils.timed
def classify_face():
    global model
    model = make_classifier(sess, graph, embeddings, classifier_filename)
    sys.stdout.flush()
    return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/classifyFace_rbf', methods=['POST'])
@face_utils.timed
def classify_face_rbf():
    global model
    model = make_classifier(sess, graph, embeddings, classifier_filename,
                            'rbf')
    return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/evolve', methods=['POST'])
@face_utils.timed
def evolve():
    global model
    model = make_classifier(sess, graph, embeddings, classifier_filename)
    return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/getIds', methods=['GET'])
@face_utils.timed
def get_ids():
    return json.dumps(list(model.classes_))


@app.route('/removeFace', methods=['DELETE'])
@face_utils.timed
def remove_faces():
    global embeddings
    embeddings = {}
    os.remove(embedding_dat_path)

    classes = os.listdir(img_dir)
    for id in classes:
        shutil.rmtree(os.path.join(img_dir, id))

    return make_response(jsonify({'ok': 'ok'}), 201)


@app.route('/removeFace/<string:id>', methods=['DELETE'])
@face_utils.timed
def remove_face(id):
    global embeddings
    if id in embeddings:
        del embeddings[id]
        with open(embedding_dat_path, 'wb') as outfile:
            pickle.dump(embeddings, outfile)
            pickle.dump(model_name, outfile)

        id_dir = os.path.join(img_dir, id)
        if os.path.exists(id_dir):
            shutil.rmtree(id_dir)
        return make_response(jsonify({'ok': 'ok'}), 201)
    return make_response(jsonify({'error': 'invalid id'}), 403)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, use_reloader=False)
