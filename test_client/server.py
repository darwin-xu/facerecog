"""Server."""
from flask import Flask, Response, request
import json
# from PIL import Image
# import numpy as np

app = Flask(__name__)


@app.route("/")
def hello():
    """hello."""
    return "Hello World!"


@app.route("/detectFace", methods=['GET', 'POST'])
def detectFace():
    """dectFace."""
    # print("request.args", request.args)
    # print("request.values", request.values)
    # print("request.form", request.form)

    data = request.form['media']
    b = bytes(data, "utf-8")

    with open("haha.jpg", "wb") as f:
        f.write(b)

    # carr = np.array([(255, 255, 255), (0, 0, 0)], dtype='uint8')
    # output = carr[np.array(map(int, list(data)))].reshape(-1, 8, 3)
    # img = Image.fromarray(output, 'RGB')
    # img.save('haha.jpg', 'JPG')

    # cmap = {'0': (255, 255, 255), '1': (0, 0, 0)}

    # output = [cmap[letter] for letter in data]
    # img = Image.new('RGB', (8, len(data)//8), "white")
    # img.putdata(output)
    # img.show()

    # f = open("haha.jpg", "wb")
    # f.write(d)
    # print("----", d)
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
    # return json.dumps(t)
    return Response(json.dumps(r), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
