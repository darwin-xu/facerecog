#!/usr/bin/env python
from flask import Flask, render_template, Response
import urllib
import requests
import json

import FaceInfo
from time import sleep

import cv2

vc = cv2.VideoCapture(0)



def gen():
    """Video streaming generator function."""
    # while True:

    rval, frame = vc.read()

    img = 't.jpg'
    cv2.imwrite(img, frame)

    url = 'http://127.0.0.1:5000/detectFace'
    files = {'media': open(img, 'rb').read()}

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
                cv2.imshow("image", pic)
                sleep(5)

                cv2.rectangle(pic,
                              # (f["x1"], f["y1"]),
                              # (f["x2"], f["y2"]),
                              (450, 200),
                              (850, 600),
                              (0, 0, 255),
                              3)
                # # TODO: Add name on top of the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pic,
                            f["name"],
                            (750, 200),
                            font,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA)

                print("show img...")
                cv2.imshow("image", pic)
                print("show img done...")

            else:
                print("Not match...")

            sleep(3)

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)
print("wait for finish...")
sleep(3)

gen()

cv2.waitKey(0)

print("relase resoure...")
vc.release()
cv2.destroyAllWindows()
