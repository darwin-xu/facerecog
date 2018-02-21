#!/usr/bin/env python
"""Client for collect camera stream."""
import json
from time import sleep
import cv2
import requests

vc = cv2.VideoCapture(0)


def gen():
    """Video streaming generator function."""
    # while True:

    rval, frame = vc.read()

    img = 't.jpg'
    cv2.imwrite(img, frame)

    url = 'http://127.0.0.1:5000/detectFace'
    files = {'media': open(img, 'rb').read()}

    # response = requests.post(url, data=files, mimetype='image/jpeg')
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

                cv2.rectangle(pic, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                              (0, 0, 255), 3)
                # Add name on top of the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pic, f["name"], (f["x2"] - 80, f["y1"] - 10), font,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                print("show img...")
                cv2.imshow("image", pic)
                print("show img done...")

            else:
                print("Not match...")

            sleep(3)

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)


gen()

if cv2.waitKey(0) & 0xFF == ord('q'):
    print("release resoure...")
    vc.release()
    cv2.destroyAllWindows()
