#!flask/bin/python
import os.path
from flask import Flask, jsonify
from test_img import load_model, test_image

app = Flask(__name__)

@app.route('/face/<string:img_path>', methods=['GET'])
def recong_face(img_path):
    # newimgpath = test_image(model, sess, class_names, pnet, rnet, onet, img_path)
    return img_path#newimgpath

@app.route('/detectFacesC', methods=['POST'])
def detect_face_c():
    return ""

@app.route('/detectFacesD', methods=['POST'])
def detect_face_d():
    return ""

@app.route('/registerFace/<string:id>', methods=['POST'])
def register_face(id):
    return ""

@app.route('/classifyFace/<string:id>', methods=['POST'])
def classify_face(id):
    return ""

@app.route('/removeFace/<string:id>', methods=['DELETE'])
def remove_face(id):
    return ""

# model, sess, class_names, pnet, rnet, onet = load_model('models/facenet/20180220-152437', '../models/my_classifier.pkl')

if __name__ == '__main__':
    app.run(debug=True)