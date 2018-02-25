#!flask/bin/python
import os.path
from flask import Flask, jsonify
from test_img import load_model, test_image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # return __file__
    return os.path.dirname(os.path.realpath(__file__))

@app.route('/exec/<string:cmd>', methods=['GET'])
def my_exec(cmd):
    # return __file__
    result = eval("os.path.dirname(os.path.realpath(__file__))")
    print (result)
    return result

@app.route('/face/<string:img_path>', methods=['GET'])
def recong_face(img_path):
    # newimgpath = test_image(model, sess, class_names, pnet, rnet, onet, img_path)
    return class_names#newimgpath

# model, sess, class_names, pnet, rnet, onet = load_model('models/facenet/20180220-152437', '../models/my_classifier.pkl')
class_names = 'fk'

if __name__ == '__main__':
    app.run(debug=True)