"""Server."""
from flask import Flask, Response, request
import json
import os

app = Flask(__name__)


@app.route("/")
def hello():
    """hello."""
    return "Hello World!"


@app.route("/registerFace/<string:id>", methods=['GET', 'POST'])
def registerFace(id):
    """registerFace."""
    img_dir = "/home/kevin/"
    print("Handling POST request...")

    result = 'result.jpg'

    file = request.files['file']
    if file:
        file.save(
            os.path.join(img_dir,
                         result))

    return Response()

@app.route("/detectFace", methods=['GET', 'POST'])
def detectFace():
    f1 = {
        "possibility": 0.8,
        "x1": 45,
        "y1": 20,
        "x2": 85,
        "y2": 60,
        "name": "Darwin"
    }
    f2 = {
        "possibility": 0.6,
        "x1": 450,
        "y1": 200,
        "x2": 850,
        "y2": 600,
        "name": "Kevin"
    }
    r = {"timestamp": 438787, "result": [f1, f2]}
    return Response(json.dumps(r), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
