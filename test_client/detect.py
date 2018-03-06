#!/usr/bin/env python
"""Client for detect face with name and indicated rectangle."""
import os
import sys
import platform
import json
import requests
import cv2

# from PIL import Image, ImageFont, ImageDraw, ImageEnhance


def detect():
    """Detect face function."""
    img = ""

    # Default interface
    url_detect = 'http://127.0.0.1:5000/detectFacesC'

    if len(sys.argv) < 2:
        print("Usage: Please provide [detect method] <fileName>.")
        print("   -c for detectFacesC")
        print("   -d for detectFacesD")
        sys.exit(1)
    else:
        for i in range(1, len(sys.argv)):
            if sys.argv[i].startswith('-c'):
                url_detect = 'http://127.0.0.1:5000/detectFacesC'
            elif sys.argv[i].startswith('-d'):
                url_detect = 'http://127.0.0.1:5000/detectFacesD'
            else:
                img = sys.argv[i]

    with open(img, "rb") as imgFile:
        encoded_image = imgFile.read()

    files = {'file': encoded_image}

    response = requests.post(url_detect, files=files)

    print(response)

    if response.ok:
        result = json.loads(response.content.decode('utf-8'))
        print(result)

        pic = cv2.imread(img, cv2.IMREAD_COLOR)
        for f in result[0]["faces"]:
            print(f)
            if (f["possibility"] < 0.8):

                cv2.rectangle(pic, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                              (0, 0, 255), 2)
                # Add name on top of the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pic, f["id"], (f["x1"], f["y2"] + 30), font, 1,
                            (255, 0, 0), 1, cv2.LINE_AA)

                print("Matched!!!")
            else:
                print("Not match...")

        cv2.imshow("image", pic)


detect()

if cv2.waitKey(0) & 0xFF == ord('q'):
    print("release resoure...")
    cv2.destroyAllWindows()
