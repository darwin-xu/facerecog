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
        "x1":20,
        "y1":20,
        "x2":140,
        "y2":100,
        "name":"kevin"
    }
    f2 = {
        "possibility":0.6,
        "x1":60,
        "y1":60,
        "x2":120,
        "y2":140,
        "name":"jason"
    }
    r = {
        "timestamp":438787,
        "result": [f1, f2]
    }
    #return json.dumps(t)
    return Response(json.dumps(r), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
