#!/usr/bin/env python
from flask import Flask, render_template, Response
import urllib
import requests
import json
import numpy

import FaceInfo
from time import sleep

import cv2

# app = Flask(__name__)
vc = cv2.VideoCapture(0)

# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index.html')


def gen():
    """Video streaming generator function."""
    # while True:

    rval, frame = vc.read()

    img = 't.jpg'
    cv2.imwrite(img, frame)
    # yield (b'--frame\r\n'
    #        b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

    url = 'http://127.0.0.1:5000/detectFace'
    files = {'media': open(img, 'rb').read()}
    # files = {'media': 'abc'}

    response = requests.post(url, data=files)

    print("get response")

    if response.ok:
        binary = response.content
        print(binary)
        result = json.loads(binary)
        print(result["result"])

        for f in result["result"]:
            print(f)
            if (f["possibility"] < 0.8):
                # write a rectangle in the current image.
                pic = cv2.imread(img, cv2.IMREAD_COLOR)
                # print(pic)
                cv2.namedWindow('image')
                # cv2.imshow("image", pic)
                cv2.rectangle(pic,
                              # (f["x1"], f["y1"]),
                              # (f["x2"], f["y2"]),
                              (10, 10),
                              (100, 100),
                              (0, 0, 255),
                              -1)
                # TODO: Add name on top of the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pic,
                            f["name"],
                            (60, 60),
                            font,
                            4,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA)

                print("show img...")
                cv2.imshow("image", pic)
                print("show img done...")

            else:
                print("Not match...")

            sleep(5)

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)


if cv2.waitKey(1) & 0xFF == ord('q'):
    pass

print("relase resoure...")
# vc.release()
# cv2.destroyAllWindows()

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, threaded=True)
gen()
