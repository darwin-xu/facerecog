from flask import Flask, Response, request
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/detectFace", methods=['GET', 'POST'])
def detectFace():
    # print("request.args", request.args)
    # print("request.values", request.values)
    # print("request.form", request.form)
    data = request.form['media']
    #print(data)
    f = open("haha.jpg", "wb")
    b = bytes(data, "utf-8")
    f.write(b)
    #f = open("haha.jpg", "wb")
    #f.write(d)
    #print("----", d)
    f1 = {
        "possibility":0.8,
        "x1":220,
        "y1":80,
        "x2":240,
        "y2":100,
        "name":"kevin"
    }
    f2 = {
        "possibility":0.6,
        "x1":120,
        "y1":40,
        "x2":160,
        "y2":80,
        "name":"jason"
    }
    r = {
        "timestamp":438787,
        "result": [f1, f2]
    }
    #return json.dumps(t)
    return Response(json.dumps(r), mimetype='application/json')
